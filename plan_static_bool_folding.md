# Plan: Static Bool Folding in Codegen (if-elimination, constant instr bool args)

Status: design draft 2026-09-25, not started.

Depends on:
* [plan_seq_unroll_head.md](plan_seq_unroll_head.md): folding happens after the peel; the peel creates the literal head copies and the `lo >= u` tail loops.
* [plan_wgmma_zero.md](plan_wgmma_zero.md): its zero-init wgmma is the first user of the constant-bool-arg test.

Optional third user: `pre_arrive` mbarrier branch elimination ([plan_mbarrier_prearrive_branch.md](plan_mbarrier_prearrive_branch.md)).
Out of scope: `exo_floor_div` / GPU floor-division semantics (dropped by the human 2026-09-25).

## Goal

In generated C, remove `if` statements whose condition codegen can decide, and let instrs see which bool args are compile-time constants.
Motivating cases after `unroll_head`:

| Where | Condition | Value |
|---|---|---|
| head copy (`iter_k` substituted by `0`) | `0 == 0`, `0 >= 1` | literal |
| tail loop `for iter_k in seq(1, …)` | `iter_k == 0`, `iter_k >= 1` | provable from loop range |
| head guard | `0 < (31 + K_cluster) / 32` | provable from `assert K_cluster > 0` |

## Analysis: `IndexRangeEnvironment`, not SMT

Use the `IndexRangeEnvironment` that `LoopIR_compiler` already maintains (`self.range_env`, built with `fast=False`).

* **Cost.** It was about 200× cheaper per query than `new_eff.Check_ExprBound` on a tiny proc (0.016 ms vs 3.4 ms, 2026-09-25).
  SMT also rebuilds context from the proc root per query (`ContextExtraction`), so it scales badly on gemm-sized kernels, and codegen asks per statement.
* **Coverage.** It already walks the lowered, post-warp-specialization tree: `add_loop_iter` runs for every `For`, including `_CodegenPar` and `CudaTasks`, and `ManagedRingBufferIdx` has a range rule.
  SMT (`new_eff`) likely doesn't understand those lowered-only nodes (untested).
* **Preconditions.** `fast=False` binary-searches SMT bounds for `size` args named in `assert`s, once per proc, so `assert K_cluster > 0` gives `K_cluster ∈ [1, ∞)`.
* **Precision** is enough: every motivating fact is a constant loop bound or a literal.
  Its known blind spot (relational facts like `i ∈ [k, k+3]`, symbolic `lo`) isn't needed here.
* **Soundness** was reviewed 2026-09-25: the interval arithmetic is sound. Fat-fingers were fixed in `exo` `52a99db1`, and `test_range_analysis.py` passes again.
* Why the environment exists at all: `simplify` (`_DoNormalize`), `fold_buffer`, and Halide-style `bounds_inference` need cheap bounds and must *produce* intervals, which SMT can't.
  So it stays regardless of this plan.

## Design

### 1. `try_fold_bool(e) -> True | False | None` in `LoopIR_compiler`

* Literals: `Const(True/False)`, and comparisons of two constant index expressions (`constant_bound` gives a point range).
* Index comparisons via `range_env.check_expr_bound`:
  * `a < b`: `True` if provable; `False` if `b <= a` provable.
  * `a <= b`: `True` if provable; `False` if `b < a` provable.
  * `>`, `>=`: swap operands.
  * `a == b`: `True` if both sides are the same point; `False` if `a < b` or `b < a` provable. (**New**: `_check_range` has no `!=` today.)
* `and` / `or` / `not`: three-valued logic.
* Anything else (non-index reads, config reads, externs) → `None`.

### 2. Static `if` elimination

In `comp_s`'s `LoopIR.If` case (never the `with`-as-`if`: check `is_if_holding_with` first, as `comp_s` already does):
* `True` → emit the body only, no braces (or braces for scoping, if needed for name hygiene);
* `False` → emit `orelse` only, or nothing;
* `None` → unchanged.

This runs in codegen on the post-warp-specialization tree, after `unroll_head`, rather than reusing LoopIR `eliminate_dead_code`.
That's because the tree has already been lowered by `cuda_backend`: managed ring-buffer consumption, warp specialization, and `_CodegenPar`.

### 3. Constant-bool instr args

* When building an `InstrNonWindowArg` for a bool-typed expr, record `try_fold_bool(e)`.
  Expose it as `arg.static_value() -> True | False | None`; the C string is unchanged, so existing instrs don't notice.
* Zero-init wgmma codegen ([plan_wgmma_zero.md](plan_wgmma_zero.md)):
  * `static_value()` is `True`/`False` → emit the literal scale-d (`add_arg(0 or 1, constraint="n")`);
  * `None` → emit the runtime form and issue the **non-constant scale-d warning**, or an error with an opt-out (open).
    The message should say the wgmma will likely serialize unless iteration 0 is peeled (`Seq(unroll_head=1)`).
  * Policy lives in the instr, not in a generic rule.

### 4. (Optional) `pre_arrive` static enable

This follows the same idea but with different inputs.
In `comp_sync_codegen_ctx`, check `range_env` for `ManagedRingBufferIdx.arg - pre_arrive >= 0`; the ring-consumption counter is non-negative by construction.
Pass the result through `SyncCodegenCtx` so the mbarrier codegen emits an `Await` without `enable`.
Details: [plan_mbarrier_prearrive_branch.md](plan_mbarrier_prearrive_branch.md).

## Tests

* Unit tests for `try_fold_bool`: literals, loop-range-provable `==` / `!=` / `<`, unprovable → `None`, and/or/not.
* Codegen: a peeled `iter_k` loop has no `if` around the head-copy zero-init or the tail-loop Await; unprovable guards survive.
* Instr: the zero-init wgmma emits a literal scale-d for constant args, and a runtime operand plus a warning for non-constant args.
* Regression: existing goldens change only where an `if` became statically decidable. Review each diff; some `if (1)` / `if (0)`-like guards in current goldens may disappear.
