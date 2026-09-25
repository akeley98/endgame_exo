# Plan: Proxy Fence Rules, Remove `cuda_temporal`

Status: design agreed (2026-09-25), not started.

Depends on: [plan_remove_vis_flags.md](plan_remove_vis_flags.md) (QualTL proxy class; single precondition set). Land together.
Evidence and caveats: [plan_sync_semantics_research.md](plan_sync_semantics_research.md).

## Goal

1. Eliminate the temporal ("write-only needs only temporal ordering") concept entirely: no `cuda_temporal` sync-tl, no temporal timeline sets, no temporal precondition sets.
2. Replace the ad-hoc proxy fence decisions in `spork/cuda_sync_state.py` with one rule based on QualTL proxy classes.
3. Keep codegen identical to CUTLASS for TMA/wgmma multicast GEMM pipelines (no proxy fences in the main loop).

## Semantic model (stated assumptions)

This is internally consistent; it is not claimed to be what NVIDIA intends (the PTX docs cannot derive most of these patterns either way).

* There is one async proxy per cluster. Synchronization purely within the async proxy (async-proxy access → sync → async-proxy access) needs no proxy fence, regardless of CTA count.
* The implicit generic-async proxy fence on completion of TMA (`cp{.reduce}.async.bulk`, PTX §9.7.10.28.2) and wgmma (§9.7.17.4) means waiting for those instructions (commit group waits, mbarrier complete-tx) never needs an explicit fence, even across CTAs.
* Generic proxy → async proxy (in either RAW, WAR, or WAW order) needs `fence.proxy.async`. NVIDIA's own `clusterlaunchcontrol.try_cancel` example (PTX §9.7.15.18) fences a generic *read* before an async *write*, so WAR is not exempt (CUDA is not Vulkan here).
* A bi-directional `fence.proxy.async` anywhere along the base causality path orders the generic/async pair (by analogy with the alias fence bullet in PTX §8.9.5). Exo always places it last (after the sync, in the awaiting CTA), which is also correct under the stricter CTA-local reading of the mixed-proxy paper.
* WAR, WAW, RAW are all treated the same by sync-check.
* Cross-cluster async→async ordering is unmodelled (Exo has no in-kernel cross-cluster synchronization; kernel boundaries are covered by CUDA API synchronization).

## Sync-tl changes

* Delete `cuda_temporal`.
* New `cuda_mbarrier_only = {cuda_mbarrier_qual}`: replaces every use of `cuda_temporal` as L1 (`Arrive(cuda_temporal)`, `Fence(cuda_temporal, ...)`).
  Used for the "data ready" Arrive after TMA loads with trailing barriers: the data travels via pending awaits, not witnessing.
* New async-only sync-tl `{cuda_mbarrier_qual, cuda_async_proxy_retired_qual}` (name TBD; the human dislikes "async proxy"; `cuda_async_in_order` sounds wrong).
  Replaces `cuda_temporal` as L2 in WAR Awaits before TMA writes, and is the L1 of the consumer's "buffer free" Arrive.
  Optionally also replaces `cuda_mbarrier_only` (the extra records it witnesses on the "data ready" Arrive are harmless); decide during implementation.
* `cuda_in_order`, `Sm80_generic`, `cuda_generic_and_async_proxy`: drop their temporal-only members (`async`, `wg0`); their remaining members become the single set.
* `generate_arrive` (mbarrier) must accept the async-only sync-tl as L1 (plain `mbarrier.arrive`).

## Proxy fence rule

Emit `fence.proxy.async` after the sync primitive (mbarrier Await, garden-variety `Fence`, `CudaClusterSync` `Fence`) iff

    L1 ∩ generic_RAM ≠ ∅  and  L2 ∩ async_RAM ≠ ∅

Otherwise do not. No CTA-count conditions.

This subsumes existing special cases in `cuda_sync_state.py`:

* `generate_await`: `cuda_temporal.implements_first(L1)` → no fence (`cuda_mbarrier_only` has no generic QualTL).
* `generate_await`: `tcgen05_commit.implements_first(L1)` → no fence (tcgen05 QualTLs are not generic).
* `add_garden_variety_or_cluster_sync`: the `cuda_temporal` and `Sm80_generic` special cases.
* Commit groups (`add_commit_group`): L1 is `wgmma_async` / `tma_to_gmem_async` / `Sm80_cp_async`. The first two have no generic QualTLs → never fence (correct by the implicit completion fence). `Sm80_cp_async_qual` is generic; its commit group currently requires L2 = `cuda_in_order` (no async), so still no fence.

Why this is consistent: a VisRecord can only gain an async-RAM QualTL via augment by an L2 containing one. That happens either (a) after a sync whose L1 contains generic RAM (fence emitted), or (b) after witnessing via an async-only L1, which only carries records that already went through (a) or a completion (implicit fence). So "async-only L1 → L2 with `cuda_in_order_ram_qual`, no fence" never skips a needed fence.

## Traced patterns

| Pattern | Sync | Fence |
|---|---|---|
| TMA load → wgmma (RAW, multicast ok) | `Arrive(cuda_mbarrier_only)` / `Await(raw, cuda_generic_and_async_proxy)` | no |
| TMA load → `ld.shared` | same | no (implicit completion fence) |
| wgmma read → TMA overwrite (WAR, multicast ok) | commit-group Await L2 ∋ async; `Arrive(async-only)` / `Await(war, async-only)` | no (matches CUTLASS) |
| TMA write → TMA write same ring slot (WAW) | same path as WAR | no |
| `ld.shared` → TMA overwrite (`tests/cuda/test_tma.py` `war`) | `Arrive(cuda_in_order)` / `Await(war, L2 ∋ async)` | **yes** (new; required per `try_cancel` example) |
| generic write → TMA store / wgmma | `Fence(cuda_in_order, cuda_generic_and_async_proxy)` | yes |
| wgmma / TMA-store read → generic write | commit-group Await L2 ∋ `cuda2` | no (implicit completion fence) |
| `cp.async` (Sm80) → wgmma | `Arrive(Sm80_cp_async)` / `Await(L2 ∋ async)` | yes |

## Known exception: SMEM free

`ChecksOnFree` requires only `cuda_in_order_ram_qual`. In persistent kernels, every `cuda_tasks` iteration frees and re-allocates SMEM,
so a generic read in task *t* followed by a TMA write in task *t+1* (e.g. `test_tma.py` `smem_x`) is accepted without a proxy fence.
This contradicts the rule above and is accepted as a pragmatic exception, to be documented.

Option if we want zero exceptions: for memory types whose mask contains async-RAM QualTLs, run the free check twice (`{cuda2}` and `{async}`).
Kernels would then end tasks with `Fence(cuda_in_order, cuda_generic_and_async_proxy)`: one fence per task (not per stage), also paid by async-only kernels.

## Code touched

* `spork/timelines.py`: sync-tl definitions (see vis-flag plan).
* `spork/cuda_sync_state.py`: `generate_arrive`, `generate_await`, `add_garden_variety_or_cluster_sync`, `add_commit_group` (`check_L2_mechanism` expectations).
* `spork/sync_check.py`: filtering / assertions for sync-tl usage (async-only sync-tl allowed where).
* Examples using `cuda_temporal` (all need rewriting; list from grep on 2026-09-24):
  * `platforms/Sm90/tk_gemm_util.py` (raw/war/wgmma_cg; also the commit-group Await L2 must contain async so the async-only war Arrive can witness)
  * `sporkbench/examples/Sm90a_tk_fa/exocc_Sm90a_tk_attn_fwd.py` (q/k/v produced/consumed) — sporkbench edits need human buy-in
  * `tests/cuda/test_tma.py`, `tests/cuda/test_cuda_sync.py` (mbarrier qual configs, `Fence(Sm80_*, cuda_temporal)`), `tests/cuda/test_3cycle_mbarrier.py`, `tests/cuda/test_ring_buffer_scheduling.py`, `tests/cuda/test_claude_*`
  * `Await(C_barrier, cuda_temporal, 0)` somewhere in tests (grep)
* Error message in `generate_arrive` ("use cuda_temporal, and add trailing barriers to TMA instrs") and the test asserting it (`test_mbarriers_wrong_tma`).
* Goldens for all of the above.

## Deferred to a future project

Memory/proxy fences should probably be modeled head-on (a fence "on the causality chain" between two accesses) rather than folded into QualTLs and sync-tl set membership.
