# Plan: Remove Visibility Flags

Status: design agreed (2026-09-25), not started.

Depends on: nothing. Should land together with (or immediately before) [plan_proxy_fence_rules.md](plan_proxy_fence_rules.md),
since that plan removes the last reason for the temporal/full distinction.
Related research record: [plan_sync_semantics_research.md](plan_sync_semantics_research.md).

## Goal

Eliminate the visibility flag (`VF_atom`, `VF_temp`, `VF_full`, `VF_issue`) from the abstract machine.
A timeline signature becomes `(thread, qual-tl)` instead of `(thread, qual-tl, visibility flag)`.
Everything the flags used to encode moves into the qualitative timeline (QualTL) namespace plus a small amount of per-QualTL class information.

Original goal was to replace `VF_temp` with a `cuda_temporal_qual` and fission extended timeline sets into temporal and full precondition sets.
**That is no longer the plan**: after the proxy-fence research, WAR/WAW/RAW are treated the same and there is only one precondition set per access
(see [plan_proxy_fence_rules.md](plan_proxy_fence_rules.md)).

## What the flags encode today (the facts being replaced)

In practice the 4 flags only ever appear in fixed bundles, and encode 3 facts:

| Fact | Old encoding | New encoding |
|---|---|---|
| Access was out-of-order: witnessable by Fence/Arrive, but does not satisfy later accesses | `VF_issue` without `VF_full` on VisRecord creation (`syncv_table.cpp`, `alloc_vis_record`, `access.is_ooo`) | Out-of-order initial QualTL, which never appears in any precondition set |
| Temporal-only visibility (non-transitive; enough for write-only mutates) | `VF_temp` without `VF_issue`, added by augment for `L2.temp` | **Removed entirely** (see proxy-fence plan) |
| Atomic-only visibility, for every thread | `G × Q_a × {VF_atom}` added on creation | `G × {atomic QualTL}` |

`VF_full` never appears without `VF_issue` (augment always adds both), so witness is effectively "`VF_issue` present".

## New QualTL attributes

Each `Qual_tl` (`spork/timelines.py`) gets two attributes.
The implementation class must be delivered to camspork (SyncEnv) at construction, as bitmasks.

### Implementation class (affects camspork optimizations)

* **in-order**: no restrictions.
* **out-of-order**: the initial QualTL of out-of-order accesses.
  * Must never appear in a precondition set (nee "extended timeline set").
  * May appear in L1 (first sync-tl), which is its purpose (being witnessed).
  * Only VisRecords created by these are eligible for the non-convergent out-of-order optimization (`gOooOpt.tex`; `syncv_table.cpp` near the `access.is_ooo || granularity == 1` require).
  * `AccessInfo.out_of_order` (`core/instr_info.py`, defaulted in `core/instr_class.py:553`) should become derived from, or asserted equal to, "initial QualTL is out-of-order class".
  * Members: `Sm80_cp_async_qual`, `tma_to_smem_async_qual`, `tma_to_gmem_async_qual`, `wgmma_async_smem_qual`, `tcgen05_smem_qual`, `wgmma_async_rmem_a_qual`.
* **atomic**: two new QualTLs, one for generic-proxy atomics (e.g. `red.global` via `Sm80.py` atomic instrs), one for TMA reduce (async proxy).
  Research ([plan_sync_semantics_research.md](plan_sync_semantics_research.md)) says mixing generic and async-proxy atomics is a data race, so these must be separate.
  * Never an initial QualTL; never in any L1 (witness), never in any L2 (augment).
  * Only in the precondition sets of atomic accesses (atomic mutate = its atomic QualTL ∪ its normal precondition set).
  * Added to new VisRecords as `G × {atomic QualTL}` (all threads), exactly like today's `atomic_qual_bits` path.
  * camspork's memoization hash (`hash_vis_record`, `max_non_atomic_tid`, `hash_bounds_for_arrive`) excludes intervals containing only atomic bits; this is only sound because atomic QualTLs never appear in L1.
  * `AtomicityInfo.qual_tl_list` becomes (effectively) a choice between the two atomic QualTLs.

### Proxy class (affects proxy fence codegen; see proxy-fence plan)

* **generic RAM**: `cuda_in_order_ram_qual`, `Sm80_cp_async_qual` (non-bulk cp.async is generic proxy in sm_90+ terminology).
* **async RAM**: `cuda_async_proxy_retired_qual`, `tma_to_smem_async_qual`, `tma_to_gmem_async_qual`, `wgmma_async_smem_qual`, `tcgen05_smem_qual` (formerly `_cuda_async_proxy_detection_quals`, deleted by [plan_mbarrier_codegen.md](plan_mbarrier_codegen.md) part 2).
* **neither**: `cuda_mbarrier_qual`, all register QualTLs (`cuda_in_order_rmem_qual`, wgmma A/D/zero, tcgen05 TMEM), CPU and stream QualTLs.
  The QualTL mask per memory type is load-bearing here: SMEM data never carries `cuda_mbarrier_qual`.

Sanity checks to add (Python, at sync-tl definition and instr registration; camspork `REQUIRE`s at SyncEnv construction):

* no L1 bits ∩ atomic mask
* no precondition bits ∩ out-of-order mask
* `is_ooo == bool(initial_bit & ooo_mask)`
* camspork's `qual_tl_mask ⊇ extended_qual_bits` require (`syncv_table.cpp`, `alloc_vis_record`) must exclude atomic bits

## Abstract machine changes

* `NewVisRecord(g*)` = `g* × {q_initial}` ∪ (`G × {q_atomic}` if atomic).
* Witness: `∃ (g, q) ∈ r.s` with `g ∈ g*`, `q ∈ L1`.
* Augment: add `g* × (L2 ∩ r.Q)`.
* Checks on read / mutate / free: every check uses the single precondition set of the access. No per-access-kind flag.
  * Read-after-write, write-after-write, write-after-read are all the same check.
  * Atomic mutates: precondition set includes the access's atomic QualTL.
* `Sync_tl` has one QualTL set (no full vs temporal). `implements_first/second` become subset tests.
* Precondition set = old extended set minus any out-of-order initial QualTL (see wgmma section below for why `rmem_a` needs special handling).

## wgmma register QualTLs

* `wgmma_async_rmem_a_qual`: out-of-order class. Initial QualTL of wgmma's A register operand; in `wgmma_async` L1 so commit-group Arrive can witness it.
* `wgmma_async_rmem_d_qual`: in-order class ("in-order" relative to other wgmma, not to the thread's other instructions). Stays in D's precondition set so wgmma→wgmma accumulation needs no sync.
* New `wgmma_rmem_fenced_qual`: in-order class, register ("neither" proxy class).
  * Appears only in precondition sets of wgmma register operands (A and D) and in `wgmma_fence_2`.
  * `wgmma_fence_2` becomes `{wgmma_rmem_fenced_qual}` (drops `wgA`, `wgD`).
  * NOT in the wait-group L2: the commit-group Await's L2 is the shared `cuda_generic_and_async_proxy`, also used by TMA commit groups and garden-variety `Fence`; adding it there would let `Fence(cuda_in_order, cuda_generic_and_async_proxy)` stand in for a missing `wgmma.fence`.
  * Must be added to `Sm90_TkRmemTileA/D` `qual_tl_dict` so it lands in their masks.
* `wgmma_zero_qual`: keeps its current role (IOU for `scale-d = 0` materialized by the next real wgmma).
  With temporal sync-tls gone it can no longer be granted as "temporal-only" by `cuda_in_order`;
  instead add `cuda_in_order_rmem_qual` to `wgmma_zero`'s precondition set. See [plan_wgmma_zero.md](plan_wgmma_zero.md) for the pre-existing hole.

Traced cases (all behave as intended):

1. CUDA writes A, `Fence(wgmma_fence_1, wgmma_fence_2)`, wgmma reads A: passes.
2. Same for D.
3. wgmma → wgmma chaining on D: passes via `wgD`.
4. wgmma writes D, commit, wait, CUDA reads D: passes.
5. Then CUDA writes D, next wgmma without fence: rejected (record replaced by `cuda1`-only record).

## tcgen05

No tcgen05 instrs are actually defined (`platforms/Sm100.py` only re-exports the dicts).
TMEM QualTLs (`cuda_tmem_qual_tl_dict`, "somewhat broken" by its own comment) use the extended set to model implicit pipelining; classify them as in-order and revisit when tcgen05 instrs exist.
(The InstrTL-vs-QualTL typo in that dict was fixed in exo `2457a95e`.)

## Code touched

Python:

* `spork/timelines.py`: `Qual_tl` class attributes; new QualTLs; `Sync_tl` single set; drop `_cuda_temporal_quals` (with proxy-fence plan); `generate_latex_table`.
* `core/memory.py`: `qual_tl_dict` interpretation (`q[0]` initial, precondition set derived), `make_qual_tl_mask`.
* `core/instr_info.py`, `core/instr_class.py`: `AtomicityInfo`, `out_of_order` derivation/check.
* `platforms/Sm80.py` (atomics), `platforms/Sm90/Sm90_tma_impl.py` (TMA reduce), `platforms/Sm90/Sm90_tk_mma_impl.py` (tile dicts).
* `spork/sync_check.py`: pass single precondition mask, atomic QualTL; drop temporal bits.
* `spork/camspork/camspork.py`: builder API.

camspork C++:

* `lib/syncv/tl_sig.hpp`: delete `QualBitsByVis`, vis flag constants; `TlSigInterval` holds one `qual_bits_t`. `TlSigIntervalListNode` shrinks 28 → 16 bytes (update the `static_assert`s).
* `lib/syncv/syncv_table.cpp` (~84 flag references; the "horror file"): `alloc_vis_record`, `union_tl_sig_interval`, `synchronizes_with`, `any/all_visible_to`, `from_L2` (delete), `AugmentVisRecordCallback`, join-threads command (already just unions bits), mutate checks (~line 2255), hash + validation (~line 2862), excut dump (~line 3030).
* `lib/syncv/syncv_table.hpp`: `SyncvAccessInfo` (stale comment mentions `vis_level_unordered` / `vis_level_full_ordered` — legacy "visibility level" wording).
* `lib/syncv/vis_record_history_log.*`: error formatting.
* `lib/program/{grammar.hpp,builder.*,exec.cpp,print.hpp,camspork_excut.*}`: drop `L2_temporal_qual_bits`; SyncEnv construction takes class bitmasks.
* camspork self-tests in `camspork.py` using hard-coded `atomic_qual_bits`.

Tests:

* ~3 goldens print `q -> atomic-only temporal full issue` (`tests/golden/cuda/test_3cycle_mbarrier/*`).
* `tests/cuda/test_claude_cuda_sync_err.py` expectations.

Docs (`spork/docs/spork_b`, read-only for now; list for the eventual doc rewrite):
`gVisFlag`, `gVisLevel`, `gVisSet`, `gTlSig`, `gVisRecord`, `gExtQualTL`, `gSyncTL` (table), `VisRecordState`, `VisRecordCreation`,
`Witness` (its "TODO is VF_full needed" becomes moot), `Augment`, `CheckVisRecordHelper`, `ChecksOnRead/Mutate/Free`,
`AccessBeforeSync`, `AccessAfterSync`, `Transitivity`, `gOooOpt`, `AtomicInstr`, `InstrTL`.

## Test plan

* All existing goldens and sync-check tests should pass unchanged except the vis-flag-printing goldens,
  *provided* the proxy-fence plan's sync-tl changes to examples land at the same time (otherwise the WAW-through-`cuda_temporal` path breaks every TMA ring pipeline).
* New unit tests for the class-mask sanity checks (each as a positive/negative `mkproc` pair, see `tests/cuda/CLAUDE.md`).
* Out-of-order soundness regression: two TMA writes to the same SMEM by the same thread with no sync must be rejected (this is the case that dropping `VF_issue` naively would silently accept).
