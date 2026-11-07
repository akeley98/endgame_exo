# Plan: Literal mbarrier Arrive Counts (drop home barrier semantics and the multicast convergence check)

**AI Disclosure: written by Claude Opus 5.5, distilled from an earlier research record (2026-10-09) and
decisions by David Zhao Akeley (2026-11-07).** Line numbers are from exo `akeley98/endgame` at `18e5d3a2`.

Depends on: [plan_camspork_test.md](plan_camspork_test.md) (trace testing, merged into `akeley98/endgame`).

## Goals

1. **Literal counts.** An mbarrier element's arrive count is what the hardware counts: one per arriving *thread*, per target CTA.
   It must reach the expected count `E` = the `mbarrier.init` count = `arrive_thread_count × len(cta_xor_list)` (`cuda_sync_state.py:323`).
2. **No home barrier *semantics* in the abstract machine.** camspork's Arrive stops reading or updating any one barrier's state on behalf of the others
   (`SyncvArrive::home_barrier` is deleted). The home index survives only as the anchor of the compact `ArriveIdx` encoding (below),
   which just enumerates the targets. Elsewhere `home_barrier_expr()` keeps meaning "this CTA's own shard": distributed memory analysis and codegen.
3. **Delete the static multicast convergence check** (`distributed_memory.py:588-613`).
4. **Test** with camspork trace testing (`spork/camspork_ref.py`), observing the counts only *indirectly*,
   through the pending awaits on records of plain (non-ring-buffer) data variables. Never probe the mbarrier itself;
   the managed ring buffer modeling that would need is a later task.

Also in scope (David Zhao Akeley): fix `test_arrive_incompatible_multicasts_positive`.
Out of scope: `amspork3/Smem.tex` (leave it alone for now); managed ring buffers in the reference;
removing the vestigial `SyncType.N` from Arrive (`Arrive(tl, 1)` syntax).
It doesn't get in the way: the new increment comes from the barrier mechanism and the thread count, never from `N`,
and camspork's Arrive already takes no `N`. Leave the existing `N == 1` checks and asserts alone.

## Semantics

### Counting unit, per barrier mechanism

| Mechanism | Increment per Arrive | Expected count `E` |
|---|---|---|
| `CudaMbarrier` (+ `PreArrive`) | number of threads in the executing collective, to **each** target barrier | `arrive_thread_count × len(cta_xor_list)` |
| Commit groups (`Sm80_cp_async`, `tma_to_gmem_async`, wgmma) | 1 per collective (thread, warp, warpgroup) | none (lag-`N` semantics unchanged) |
| `CudaClusterSync` | 1 per collective | 1 (replaces `one_shot_arrive`; `one_shot_await` stays) |

This matches amspork3, which leaves the arrive count increment to the barrier mechanism's de-sugaring (`SyncTraceSemantics.tex`, note on `Tr-Arrive`).
`BarrierMechanismTraits.one_shot_arrive` and camspork's `one_shot_arrive_flag` are deleted. They're replaced by
a per-barrier `expected_arrive_count` (0 = unchecked) and an `arrive_per_thread` flag, both set at `BarrierEnvAlloc`.

### Arrive (`on_arrive`)

For each distinct target barrier `b` (the union of all `>>` expressions, de-duplicated by camspork):
record `PendingAwait(b, count_b)` with `b`'s **own** count before the increment, then `count_b += inc`.
If `E` is set and `count_b > E`: error ("Too many arrives").
There is no home: the history logger (`history_set_sync_stmt_info`) records before/after counts for every target.

### Await (`on_await`): "strict Await"

If `E` is set, the Await requires `count == E` at that point in the trace, or it fails with "No forward progress".
`max_arrive_count = count - 1 - N` is unchanged (`N == 0` for mbarriers).

Today's check is `count != 0` on the awaited barrier, and the stand-in count only ever reflects the awaiting CTA's own arrive.
So an Await can pass before its peer CTAs' arrives appear in the trace, and the convergence check is what hides that hole (counterexample below).
Strict Await is the hardware rule: the phase completes only when all `E` arrives have landed.

It's conservative the same way as the non-multicast case. The trace runs `cuda_threads` iterations serially,
so an Await that waits on a peer CTA's arrive *from the same iteration of the same loop nest* fails, even though it's valid CUDA.
Such programs need the Await fissioned into a later loop nest, or waiting on an arrive from an earlier ring-buffer iteration.

### Free (`free_barriers`)

If `E` is set: `count == E`. This replaces "arrive count must be 1 (one-shot)".

### Pre-arrive

sync_check currently emits one `b.Arrive(0, bar[idx], ...)` per pre-arrived element (`sync_check.py:426-437`).
Replace it with an operation that sets the count to `E` (requires count 0, creates no pending awaits).
For example a `PreArrive` camspork statement, or a flag on the Arrive node.

### Trailing barrier expressions

Delete the check in the `!ForceArriveCount0` branch of `alloc_vis_record` (`syncv_table.cpp:788-797`),
which demands that no Arrive came before the access ("one-shot arrive requirement").
Each trailing barrier's PendingAwait gets that barrier's literal current count.

**Acknowledged gap:** Exo doesn't check the ordering of the `expect_tx` Arrive against the TMA's `complete_tx`, or the byte accounting.
That check was always unsound. amspork3 doesn't document it either.

The `ForceArriveCount0 = true` path (managed ring buffer alloc records, `set_managed_ring_buffer_barriers`) is unchanged here and belongs to the ring buffer task.

### Encoding: keep the compact `ArriveIdx` (anchor + per-expression multicast flags)

Decision (David Zhao Akeley, 2026-11-07): **don't change the encoding in this plan.**
`ArriveIdx` / `eval_multicast` is shared with `SyncEnvManageRingBuffer`, which is untested and not yet specified (see the next section).
So sync_check still computes the anchor with `home_barrier_expr()` for `b.Arrive` and the barrier `SyncEnvAccess`, but it's only an encoding:
`eval_multicast` turns (anchor, flags) into the target set, visiting each position once, so the target list has no duplicates.
`fill_tmp_offset_barriers` stops returning the home barrier, and `on_arrive` sees only the target list.
The Await may use `s.barriers[0]` instead of `home_barrier_expr()` (the same thing for a single point expression; optional).

A per-expression encoding, where points may differ between expressions and no anchor exists, can come with the managed ring buffer task,
once `SyncEnvManageRingBuffer` is specified and tested.

### `SyncEnvManageRingBuffer`'s use of the encoding (what the code does today; not a spec)

From `exec.cpp` `exec_impl(const SyncEnvManageRingBuffer*)` and `sync_check.py:439-505`, for each guard position `i` of buffer `x` (guard `g`, ring dimension `M`, depth `d`):

* **Self-guarded** (a barrier ring, `g == x`): the multicast flags are `()`, so the anchor's own position is the only target.
  alloc_on_await = `{g[i - d]}` (none if `i[M] < d`).
* **Guarded data buffer:** the flags are the guard's Arrive multicasts (`DistributedAllocState.barrier_multicasts`, the same for all Arrives on `g` by `BarrierUsage`).
  alloc_on_await = every guard position that **an Arrive anchored at `g[i]` would multicast to**, i.e. the anchor is `i` itself, not `i - d`.
  This is a second, implicit home-style use: it treats `i` as the home of a hypothetical Arrive.
* free_on_arrive = `g[i + d]` only, never multicast (none if self-guarded and out of range).
* The alloc record uses `ForceArriveCount0`: `PendingAwait(b, 0)` for each alloc_on_await barrier `b`.

Interaction with this plan: none intended. With literal counts and strict Await, `z = 0 ≤ E - 1`, so any Await that passes still covers the alloc record.
Mismatch to resolve in the ring buffer task: `plan_camspork_test.md`'s draft `allocTrDetail` uses the single barrier `g[..i-d..]`,
while the code uses `g[i]` plus its multicast peers for guarded data buffers.

## Implementation steps

1. **Python-side `E`.** Factor the expected count out of `cuda_sync_state.py:296-323`
   (`arrive_thread_count × len(cta_xor_list(...))`) into one helper that both codegen and sync_check call.
   Check that sync_check has what the helper needs (CollAnalysis thread iters, blockDim, the barrier usage's arrive `SyncInfo`); if it doesn't, thread it through.
   Pass `E` and `arrive_per_thread` to `BarrierEnvAlloc`; delete the `one_shot_arrive` trait and flag.
2. **camspork** (`syncv_table.{hpp,cpp}`, `exec.cpp`, `camspork.py`):
   * `HamsterBarrierState` gains `expected_arrive_count` and the per-thread flag.
   * `on_arrive`: per-target counts and the increment (thread count of the cuboid, or 1).
   * `on_await`: strict check. `free_barriers`: `count == E`.
   * Pre-arrive op.
   * Remove the trailing-barrier one-shot check.
   * `fill_tmp_offset_barriers` no longer returns the home; `SyncvArrive::home_barrier` is deleted. `ArriveIdx`, `eval_multicast` and `SyncEnvManageRingBuffer` are untouched.
   * The history logger is called per target barrier.
   * Keep `syncv_table.cpp` edits to these places.
3. **sync_check** (`sync_check.py:251-300`): the Arrive lowering is unchanged apart from comments calling the home index the encoding anchor.
   Optionally, the Await uses `s.barriers[0]`.
4. **Reference** (`camspork_ref.py`, class `Barriers`): `alloc(name, expected=None, per_thread=False)`,
   `arrive(targets, n_threads)` with per-target counts, plus the strict Await, free and pre-arrive checks.
   Delete `arrive_multicast_mbarrier` and the one-shot arrive flag.
5. **Audit before deleting the convergence check.** With steps 1-4 in, run every sync_check that uses multicast mbarriers:
   `tests/cuda/test_sync_error_Sm90a_gemm.py`, the Sm90 sporkbench examples (compile + sync_check only on this machine), and `tests/cuda`.
   The expectation is that the GEMMs pass, because their multicast Awaits (consumer "buffer free" barrier) wait on arrives from an earlier ring-buffer iteration.
   **If any kernel fails "No forward progress" under strict Await, stop and report it** before changing the schedule or sporkbench.
6. **Delete the convergence check** (`distributed_memory.py:588-613`, the `if any_multicast:` block).
   No existing test covers it (grep for "convergence requirement" finds only the source).
   The interval range check above it stays.
7. **Tests** (below), then the full `tests/cuda` run per `exo/CLAUDE.md`.
8. **Doc notes** (spork_b is read-only): record in this file that `BarrierMulticast.tex` (convergence requirement), `ArriveSemantics.tex` (home count),
   `ArriveAwaitPairing.tex` ("batches"), `gHomeBarrier.tex` (semantic role) and `MbarrierUsage.tex` (one-shot arrive) no longer match.

## Testing (`tests/cuda/test_camspork_trace.py`)

Shared setup:
* Sm90 cluster programs; sync_check runs on CPU.
* Every mbarrier has extent == ring depth (`barrier[2, K @ ring_buffer_by(K)]`), so ring slots are never reused.
* Probes are plain GMEM data variables only. The reference models mbarrier counts but not the barrier variable's own sync env.
* Hand-compute `E` and the thread counts in each `mkref_*`, like the tids.

How the counts show up: a data record written by CTA `c` and then witnessed by its Arrive gets `aw = {b: count_b before}` for every target `b`.
An Await adds to the record (Ob grows) only if the strict check passes and `z ≤ E - 1`.

### Positive (full trace comparison)

* **2-CTA multicast, per-thread counts.** clusterDim=2, one warp per CTA arrives (T=32).
  * Each CTA writes `g[c]`, then `Arrive(cuda_in_order) >> bar[c, 0] >> bar[:, 0]`.
  * In a later loop nest, each CTA does `Await(bar[c, 0])` and reads `g[1 - c]`.
  * Expected: `g[0].aw = {bar[0,0]: 0, bar[1,0]: 0}` and `g[1].aw = {bar[0,0]: 32, bar[1,0]: 32}`. Today's home rule gives `{0, 0}` for `g[1]`.
  * Both Awaits pass with `E = 64`.
* **Collective size variant:** the same with a 2-warp arrive collective (T=64, E=128), so `z` is visibly per-thread.
* **4-CTA, 2-D multicast:** `Arrive >> bar[m, n, k] >> bar[m, :, k]` (group of 2 along `n`).
  The counts grow only within row `m`; `bar[m', ...]` is untouched.
* **`seq k` loop inside the CTA loop, Awaits in a later loop nest.** The convergence check used to reject this statically; it's valid CUDA.
  Check `z` per `k`.
* **Pre-arrive:** an Await on a pre-arrived element passes and augments nothing tied to that element.
* **Trailing barrier after a peer's Arrive** (only if a non-ring-buffer SMEM TMA destination passes sync_check; otherwise defer to the ring buffer task):
  the record's PendingAwait carries the literal count, and the Await covers it.
* Commit-group and cluster-sync counting are already covered (`test_cp_async_commit_group`, `test_cluster_sync`); they must keep passing unchanged.

### Negative (both camspork and the reference reject; the trace is ignored)

* **Turnstile, `lag=2`** (deadlocks in CUDA) and **`lag=1`** (valid CUDA, but the serialized trace can't show it):

```python
# clusterDim=2, blockDim=32, K=4
bar: barrier[2, K @ ring_buffer_by(K)] @ CudaMbarrier
tb: barrier[2, (K + lag) @ ring_buffer_by(K + lag)] @ CudaMbarrier  # turnstile
for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
    if cta == 1:
        for j in seq(0, lag):
            Arrive(cuda_in_order) >> tb[cta, j] >> tb[:, j]
    for k in seq(0, K):
        Arrive(cuda_in_order) >> bar[cta, k] >> bar[:, k]
        if cta == 0:
            Arrive(cuda_in_order) >> tb[cta, k] >> tb[:, k]
        else:  # CTA 1 at iteration k waits for CTA 0 to reach iteration k + lag
            Arrive(cuda_in_order) >> tb[cta, k + lag] >> tb[:, k + lag]
            Await(tb[cta, k + lag], cuda_in_order, 0)
        Await(bar[cta, k], cuda_in_order, 0)
    if cta == 0:
        for j in seq(K, K + lag):
            Arrive(cuda_in_order) >> tb[cta, j] >> tb[:, j]
```

  Today both pass sync_check if the convergence check is disabled. Under literal counts, CTA 0's first `Await(bar[0, 0])` sees 32 of 64 arrives.
* `if n_cta == 0:` around a multicast Arrive: a missing peer arrive fails the strict Await or the free check.
* An Await in the same CTA body as its multicast Arrive (`Arrive >> bar[m,n,0] >> bar[m,:,0]; Await(bar[m,n,0])`): "No forward progress" (the documented conservatism).
* Two Arrives on the same element: "Too many arrives".
* A partial arrive never completed before free: the free check.

### Existing tests

* `test_arrive_incompatible_multicasts_positive` (`tests/cuda/test_claude_barrier_usage_err.py:334`) is invalid CUDA:
  its second Await reuses parity 0 (2 arrives, ring depth 1). Give each Arrive its own ring element, with Awaits that pass strict Await,
  and actually run sync_check on it (`cuda_cpu_test` doesn't).
* `test_sync_error_Sm90a_gemm.py` and the golden tests must pass unchanged (this plan changes no codegen or printing).

## Implementation status (2026-11-07, exo `akeley98/literal_mbarrier_counts`, uncommitted)

Steps 1-7 are implemented; step 8 (doc notes) is the list at the end of this section.
`tests/cuda`: the full run had 5 failures, all fixed afterwards (below), and the affected files pass on re-run.
The full suite was not re-run after those test/golden-only fixes.

**Audit (step 5):** all 36 Sm90 sporkbench scripts (60 sync_check calls: handwrite/schedule GEMMs incl. m1n2, m2n1, ping-pong, split-K;
row-major GEMMs; tk_fa; unflash_attn) pass under strict Await, run from a scratch copy so sporkbench's `.json` outputs weren't touched.
So no real kernel relied on an Await seeing a peer's arrive from the same loop iteration.

**Tests added** (`tests/cuda/test_camspork_trace.py`, all passing):
* positive trace comparisons: 2-CTA multicast with 1 and 2 arriving warps (`z = 32` / `64` on CTA 1's records, where the home rule gave 0),
  4-CTA multicast along `n` only, pre-arrive, and a `seq` loop between the CTA loop and the Arrive (previously rejected statically);
* negatives (both reject): Await in the same CTA body as its multicast Arrive, two Arrives on one element ("Too many arrives: 96 exceeds 64"),
  `if cta == 0:` around the Arrive, and the turnstile with `lag=1` and `lag=2`.
  All of these fail in camspork with the intended message ("arrive count 32 of expected 64" etc.), checked by hand.
  Before the convergence check was deleted, the last three failed statically, so `check_both_fail` would have passed for the wrong reason.
* `test_arrive_incompatible_multicasts_positive`: one ring element per Arrive, Awaits in a later loop nest, and now actually sync-checked.

**Complications found:**
1. **Pre-arrive needed a camspork grammar change.** sync_check issues pre-arrives from the allocating collective (the whole cluster),
   so a per-thread increment would be wrong. `Arrive` gained a `flags` field (`arrive_pre_arrive_flag`: count := E, no PendingAwaits).
   It's still traced as an Arrive event by the allocating threads, and the reference's `pre_arrive` does the same.
2. **History log holds one barrier per sync statement** (`set_syncv_sync_stmt_info` is state for later log calls).
   With no home, it logs the first target's counts. Error messages for multicast Arrives show only one barrier.
3. **Goldens:** the 4 `test_3cycle_mbarrier` error-remark goldens change, only in logged counts (`0 -> 1` becomes `0 -> 32`, `<= 0` becomes `<= 31`, etc.).
   Pytest's truncated diff showed only a source-path line, which is misleading; regenerate with `--update-golden` to see the real diff.
4. **`test_ping_pong_reorder_arrive_raw`** (producer's `raw` Arrive moved before its TMA) expected the deleted trailing-barrier message.
   It's still rejected, but later: the TMA's trailing PendingAwait gets the already-complete count (`raw[0, 0, 0, 0] arrive_count=64`),
   which no Await covers, so the consumer's read of `B_smem` is a hazard (named `use_before_alloc`). The test now checks for that.
   Open question: with `expect_tx` this order (arrive, then TMA) may be valid CUDA, as in CUTLASS; if so, this is the acknowledged expect-tx gap, rejecting conservatively.
5. **`CudaClusterSync`** now reports "Too many arrives" instead of "Arrived twice on one-shot arrive barrier"; no test depended on the old text.
6. Running sporkbench scripts or sync_check outside pytest/exocc needs `CAMSPORK_JIT_DIR` (environment only).

**Not done:** a positive trace test for a multicast trailing barrier (TMA) after a peer's Arrive.
That needs a TMA destination that sync_check accepts without a managed ring buffer; deferred to the ring buffer task.

**Doc notes (step 8; spork_b is read-only, not caused by unrelated drift):** these no longer match the implementation:
* `BarrierMulticast.tex`: the convergence requirement is gone.
* `ArriveSemantics.tex`: only the home count was used; now each target's own count is.
* `ArriveAwaitPairing.tex`: "batches" of B arrives incrementing each count by 1; now per-thread increments to an expected count.
* `gHomeBarrier.tex`: the home barrier has no semantic role in sync-check, only as an encoding anchor, in distributed memory analysis, and in codegen.
* `MbarrierUsage.tex`: "one-shot arrive" is replaced by "arrive count reaches the mbarrier.init count".

## Background: why literal counts must land before the convergence check goes

* Today `on_arrive` (`syncv_table.cpp:1953-1976`) increments only the home barrier's count and tags every target's PendingAwait with it.
* That is count-correct only under one-shot plus the distributed memory analysis (each element is the home of exactly one CTA).
* Forward progress is **not** checked: `on_await` only asks whether the awaiting CTA's own arrive happened.
* The static convergence check is what currently rules out programs like the turnstile `lag=2` example, which pass sync_check without it and deadlock in CUDA.
