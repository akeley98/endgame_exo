# Plan: `Seq(unroll_head=...)` Loop Mode

Status: idea agreed 2026-09-25; design draft, not started.

Depends on: nothing.
Enables: [plan_static_bool_folding.md](plan_static_bool_folding.md) (the peel creates the facts that fold), removing `cut_loop` from `tk_gemm_util.sched_cut_sync_iter_k`, and the `pre_arrive` branch elimination ([plan_mbarrier_prearrive_branch.md](plan_mbarrier_prearrive_branch.md)).
Related: [plan_wgmma_zero.md](plan_wgmma_zero.md); `cuda_multi_for` / `seq` Unrolling in [CLAUDE.md](CLAUDE.md) (**unresolved wrench**, see below).

## Motivation

* ptxas does not peel iteration 0 itself.
  A runtime `iter_k == 0` scale-d (or `if iter_k == 0: mm else: mma`) in an unpeeled wgmma loop gets C7515 "wgmma serialized", even with `#pragma unroll`.
  A peel in the generated C fixes it (experiments in [plan_wgmma_zero_alternatives.md](plan_wgmma_zero_alternatives.md)).
* Today the peel is `cut_loop(iter_k, 1)` in LoopIR, which also duplicates the (identical) producer code and complicates every later schedule step.
* `pre_arrive` branch elimination needs a producer loop that starts at `ring_depth`, and nvcc won't derive it ([plan_mbarrier_prearrive_branch.md](plan_mbarrier_prearrive_branch.md)).

## Semantics

`Seq(unroll_head=u)` (also `seq(lo, hi, unroll_head=u)`) is a codegen annotation. It does not change program meaning:

    for i in seq(lo, hi, unroll_head=u): s
    ==>
    if lo + 0 < hi: s[i -> lo + 0]
    if lo + 1 < hi: s[i -> lo + 1]
    ...
    if lo + (u-1) < hi: s[i -> lo + (u-1)]
    for i in seq(lo + u, hi): s          # keeps any pragma_unroll

`s[i -> n]` substitutes `n` for `i` and alpha-renames the copy (fresh Syms for inner loop iterators, allocs and window statements; `Alpha_Rename` exists).
The guards are ordinary LoopIR `If`s. [plan_static_bool_folding.md](plan_static_bool_folding.md) removes the provable ones, e.g. `lo + 0 < hi` given `assert K_cluster > 0`.

Only `Seq` loops get the option; `cuda_tasks`, `cuda_threads` and `par` loops don't.

## Implementation: LoopIR → LoopIR pass in codegen

* Place it in `backend/LoopIR_compiler.py` right after the `"scheduled"` debug-log point, before `ParallelAnalysis` … `BarrierUsageAnalysis` / `CollAnalysis` and before `loopir_lower_cuda`.
  Every later stage then sees the peeled code.
* That output shape (head copies + tail loop) is exactly what `cut_loop` + `unroll_loop` produce today.
  So barrier usage analysis, distributed-memory deduction, `cuda_backend` lowering, and ring-buffer consumption lowering should already handle it.
  Verify this rather than assume it.
* sync-check (`Procedure.sync_check`, camspork) runs on the un-peeled proc and treats `unroll_head` as a no-op annotation, since the pass is semantics-preserving.
* Claude: The annotation has to survive scheduling ops that copy or rebuild loop modes (`set_loop_mode`, `update_loop_mode`, `divide_loop`, `cut_loop`, `fuse`, …).
  Decide per op whether it keeps, drops or rejects it; the default could be to drop it on any op that changes the loop bounds.
  David Zhao Akeley: for `set_loop_mode` the wholesale replacement of the loop mode is intended.
  `update_loop_mode` should preserve the annotation via `LoopMode.update`.
  The other functions probably preserve the annotation as well by default
  due to the ADT `update` function ... I'm not too concerned about this behavior,
  since setting the loop mode tends to be a scheduling "finishing touch" anyway.

## Iteration-specific sync without `cut_loop`

`sched_cut_sync_iter_k` uses the cut for a second reason: the wgmma-retire `Await(wgmma_cg, 1)` and the `war` `Arrive` exist only for `iter_k >= 1`.
Without the cut:

    p = insert_await(p, gap, "wgmma_cg[cta_m, cta_n, wg_m]", cuda_in_order, 1)
    p = add_if(p, <the Await/Arrive block>, "iter_k >= 1", unsafe_disable_check=True)
    p = set_loop_mode(p, iter_k_loop, Seq(unroll_head=1))

* `add_if` wraps the whole block. It requires `unsafe_disable_check=True` (the safe check is `assert ... "not implemented"`); accepted as-is to avoid scope creep.
* Per the human, sync-check handles `if`-guarded sync statements.
* After the peel, `iter_k >= 1` is literally false in the head copy and provably true in the tail loop. [plan_static_bool_folding.md](plan_static_bool_folding.md) removes both.

## Wrench: producer vs consumer head counts (`cuda_multi_for`)

The consumer wants `unroll_head=1` (wgmma zero-init), but the producer wants `unroll_head=ring_depth`, so its steady-state loop starts at `ring_depth` for the pre-arrive analysis.
Both live in one `iter_k` loop split by `CudaWarps`.
A single-int `unroll_head` means either:
* the consumer also peels `ring_depth` copies (code size), or
* the producer peels just 1 (no pre-arrive gain).

Options, not decided:
* `unroll_head={"producer": 4, "consumer": 1}`, keyed by warp name. The pass would then have to run after warp specialization, which conflicts with running it before the analyses above.
* Fold it into `cuda_multi_for`'s per-warp `Seq` modes, if `cuda_multi_for` ever exists.
* Accept a single int for now: use 1 for the wgmma zero-init, and leave the pre-arrive work waiting.

## Tests

* Codegen: head copies contain the loop variable as a literal, and the tail loop starts at `lo + u`.
* Guards: head guards are present when unprovable, and absent once [plan_static_bool_folding.md](plan_static_bool_folding.md) proves them.
* Runtime (sm_80 box): a CPU/`Sm80` proc with `unroll_head` gives identical results for trip counts `< u`, `== u` and `> u`, including 0.
* The excut line-number idea from the `seq` Unrolling section of [CLAUDE.md](CLAUDE.md) can confirm which copy executed.
* Sm90 gemm (compile-only here): replacing `cut_loop` in `sched_cut_sync_iter_k` with `add_if` + `unroll_head=1` still passes sync-check, and `-Xptxas -v` shows no C7515.

## Rejected sync-check test (Human Addition)

A previous agent wrote "Open: whether to also sync-check the peeled
form in tests as a cross-check.".  I think this is not feasible
because the real deeper purpose of this is to prepare for the
`cuda_multi_for` change, and that's predicated on the
post-unroll-head-transformed code *failing* sync-check (due to "no
forward progress" or related issues).

## More Human Addition

Actually I wonder if the whole LoopIR→LoopIR rewrite can be done "just
in time" in the compile-Seq-loop code. This will completely bypass the
ordering problems with the hypothetical `cuda_multi_for` loop.  Also,
there's a serious foot-gun in the proposal where `sync_check.py` takes
as input a LoopIR nest and a `CollAnalysis` *that is generated from a
LoopIR nest considerably structurally different*.
