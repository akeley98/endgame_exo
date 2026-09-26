# Research: `pre_arrive` mbarrier branch elimination

Status: research only (2026-09-25).
Justification for the approach in `CLAUDE.md`, "`mbarrier` Branch Elimination": do the analysis in the Exo compiler, because nvcc won't do it.
Related: [plan_mbarrier_codegen.md](plan_mbarrier_codegen.md).
Implementation path: [plan_seq_unroll_head.md](plan_seq_unroll_head.md) (producer loop starting at `ring_depth`) and [plan_static_bool_folding.md](plan_static_bool_folding.md) §4.

## Does cutting the producer loop remove the branch?


**No.** Tested on the scheduled f32 m1n2 gemm (`schedule_gemm`, ring_depth 4).
The source got `assert K_cluster > 96` (so `cut_loop` accepts it), and the `iter_k >= 1` loop was cut again at 4, giving `seq(0,1)`, `seq(1,4)`, `seq(4, …)`.
Sync-check passes.
In SASS, the war wait in the `iter_k >= 4` producer loop is still `ISETP.GE P0, (iter_k + rc), 4; @!P0 BRA` around the `SYNCS.PHASECHK.TRANS64.TRYWAIT` (same as every other copy).

Why: the Await index is `iter_k + exo_syncState.ring_consumption_1_war - ring_depth`, and `enable = i0 >= 0`.

* `ring_consumption_*` (`rc`) is a runtime counter carried across persistent-kernel tasks.
  The skip only matters on each CTA's first task; afterwards `rc >= K_iters + ...`, and every iteration (even `iter_k < ring_depth`) must wait.
  So the branch can never be removed for `iter_k < ring_depth` without also peeling the first task. It *can* be removed for `iter_k >= ring_depth`, if `rc >= 0` is known.
* `rc` is reassigned at task end via `__shfl_sync` (deliberate: forces uniform registers), which is opaque to the compiler.
  Its value is also computed in `int_fast32_t` (64-bit on device) and narrowed to `int` (implementation-defined, not UB).
  So nvcc has no range info.
* Hand-edited `.cuh` experiments (same SASS check):
  * `__builtin_assume(rc >= 0)` after each `__shfl_sync` update: **byte-identical SASS** (the assumption doesn't survive the task-loop back-edge).
  * `__builtin_assume(rc >= 0)` right before each `Await0_war(...)` call: the test becomes `ISETP.GE.U32`, so nvcc now knows both operands are non-negative, but **the branch stays**.
  * Additionally `__builtin_assume(iter_k >= 4)` at the call in the `seq(4, …)` loop: **branch still stays**.
    nvcc/LLVM won't derive `iter_k + rc >= 4` from `iter_k >= 4 ∧ rc >= 0`.

Conclusion: downstream compilers won't remove it even with perfect hints.
Exo must decide statically: e.g. the `LoopIR_compiler` / `cuda_sync_state` codegen proves `iter_k - ring_depth >= 0` from the loop bounds (the `rc >= 0` part is known by construction), then emits the Await with `enable = true` (a no-check variant).
This needs the `seq(ring_depth, …)` producer loop to exist, via `cut_loop` or a future consumer/producer-specific peel (see `cuda_multi_for` in `CLAUDE.md`).

Unsigned-math audit of the `enable` chain (`i0 = int(iter_k + rc - ring_depth)`; `i0 >= 0`): **all signed `int`**, so signed-overflow UB is available to the optimizer.
The unsigned math present is off the chain:
* `unsigned(i0) % ring_depth` (slot) and `(unsigned(i0) / ring_depth) & 1` (parity)
* `blockIdx.x % 2` (`cta_rank`)
* `exo_TaskGenerator`'s `uint32_t` task index/count

Side observation: `size` args are `int_fast32_t`, which is 64-bit on device.
So loop bounds like `(31 + K_cluster) / 32` are 64-bit and produce `ISETP … .EX` pairs; this is a separate micro-inefficiency.

## Sketch of the Exo-side analysis

At each `Await` on a `CudaMbarrierPreArrive` barrier, codegen emits the index `iter_k + rc - ring_depth` (with the barrier's actual index expression in place of `iter_k`).
It can drop `enable` when:

* `rc >= 0`: always true by construction (starts at 0, only increases). Codegen owns this counter, so no proof is needed.
* `index_expr - ring_depth >= 0`: needs a lower bound on the loop variables in the (affine) index expression.
  For `for iter_k in seq(4, …)` with `ring_depth = 4`, this is just the loop's `lo`; existing range analysis should cover it.

When both hold, emit a no-check `Await` variant (or pass a constant `enable = true`).
For `iter_k < ring_depth` the branch is a genuine runtime decision (skip on a CTA's first task, wait on later tasks) and must stay.
The payoff depends on a producer loop starting at `ring_depth` existing, via `cut_loop` today or a future producer-only peel (`cuda_multi_for`).
