# Research Record: Multicast mbarrier Counting (home barrier, Arrive N, convergence)

Not yet an implementation plan. Collected 2026-10-09.
Question: can multicast `Arrive` semantics drop the static convergence requirement (`spork_b/BarrierMulticast.tex`),
e.g. by modeling each mbarrier's arrive count as the literal CUDA count and replacing "one-shot arrive" with "arrive count = expected arrive count"?

## (a) Where the home barrier expression is used

`LoopIR.SyncStmt.home_barrier_expr()` (`exo/src/exo/core/LoopIR.py:914`) merges the `>>` barrier exprs: same name, every dimension has at least one point, and all points agree.

| Use | Location | Role |
|---|---|---|
| Abstract machine Arrive | `camspork/lib/syncv/syncv_table.cpp` `on_arrive` (~l.1951); `program/exec.cpp` `fill_tmp_offset_barriers` | **Semantic.** Only the home's `arrive_count` is read and incremented. Every referenced barrier gets a `PendingAwait` tagged with the *home's* old count. The one-shot check looks only at the home. |
| sync_check lowering | `spork/sync_check.py:245-290` | Computes the home index to pass to `b.Arrive`. The barrier `SyncEnvAccess` is at the home with `access_multicasts`, so it still covers every referenced barrier. |
| Distributed memory analysis | `spork/coll_analysis.py:268`, `spork/distributed_memory.py:575` (`consume_SyncStmt_idx`) | `DistributedIdxFsm` parses the home expr: distributed indices must be the executing CTA's iterators. **The convergence check lives here** (l.599-623). |
| Codegen | `backend/compiler_fwd.py:43` (`cir_raw_non_distributed_home_barrier_idx`), `backend/LoopIR_compiler.py:1328-1377`, `spork/cuda_sync_state.py:517` | The home's non-distributed indices give the local mbarrier. Peers are reached with `mapa(cta_rank ^ xor)` over `cta_xor_list`. |
| Validation | `spork/barrier_usage.py:71,122` | Name and shape checks only. |
| Cursor API | `API_cursors.py:550` | |

Outside camspork's `on_arrive`, "home" only means **the executing CTA's own shard**, which is a syntactic and codegen notion.
`on_arrive` is the one place where it has semantic weight, as the stand-in arrive count that `ArriveSemantics.tex` describes.

## (b) `SyncType.N` for Arrive types is vestigial

* `pyparser.py:2008-2020`: accepts `Arrive(tl)` or `Arrive(tl, 1)` and raises `"Expect N=1 for Arrive"` otherwise.
* `arrive_type(tl, N=1)` is the default, and `DoInsertArrive` (`LoopIR_scheduling.py:2915`) hardcodes `1`.
* `BarrierUsage.visit_Arrive` enforces `N == 1` again. It records `min_N`/`max_N`, but nothing ever reads them, for Arrive or for Await.
* `cuda_sync_state.py:525,621` asserts `N == 1` (mbarrier, commit group).
* The camspork `b.Arrive(...)` takes no `N` at all. Only `Await` passes `N` (`sync_check.py:289`).
* Fence decomposition stores Arrive `N=1` / Await `N=0` in `SyncInfo`.
* Printing: `format_stmt` prints `Arrive(tl, 1)`, while `__str__` prints `arrive_type(tl)`.
* Latent bug: `barrier_usage.py:94,144` do `info.stmts.append(info)` instead of `append(s)`. It is harmless today because only `stmts[0]` is read.

Options: delete the field for Arrive, or repurpose it as the PTX `mbarrier.arrive` count operand if a use appears. The only effect of deleting it is that `Arrive(tl, 1)` syntax goes away, and goldens and examples print it.

## Motivation for literal counts

Primary: the abstract machine should model CUDA reality directly.
An mbarrier's arrive count is the number of arrives it has physically received, and it must reach the `mbarrier.init` count.
That is easy to explain. Today's spec instead updates only the home barrier's count as a stand-in for the whole group.
Showing that the stand-in is even count-correct takes a cardinality argument (below), which is hard to follow.

Secondary: today's model does not check forward progress for multicast barriers.
Its Await check (`on_await`: home `arrive_count != 0`) only asks whether *the awaiting CTA's own* arrive has happened, not whether its peers' have.
Today, the static convergence requirement is what hides this hole.

## What one-shot does and does not give you

Count correctness holds under one-shot (CudaMbarrier, the only mechanism with `supports_arrive_multicast`):

1. One-shot means every element's home count is exactly 1 at free (`syncv_table.cpp:1135`). So every Arrive `PendingAwait` has `arrive_count == 0`.
   With `N == 0` forced, an Await at count 1 makes every pending await recorded so far eligible.
2. Each element `z[a,b,k]` is the home of exactly one CTA, `(a,b)`, because of distributed memory analysis.
   Multicast groups are symmetric, and `BarrierUsage` forces identical multicasts on all Arrives.
3. So each element physically receives exactly `E = |group| = len(cta_xor_list)` arrives, which matches `mbarrier.init` (`arrive_thread_count × E`).
   Edges are sound: every arrive that comes earlier in the trace lands in the phase the Await observes.

Forward progress does **not** hold without the convergence check.
The trace runs `cuda_threads` iterations serially (`exec.cpp` `ThreadsFor`), so an Await can pass while its peers' arrives come later in the trace.
Counterexample (sync_check with the convergence check monkeypatched off: `distributed_memory.is_if_holding_with = lambda *a: True`):

```python
# clusterDim=2, blockDim=32, K=4
bar: barrier[2, K @ ring_buffer_by(K)] @ CudaMbarrier
tb: barrier[2, (K + lag) @ ring_buffer_by(K + lag)] @ CudaMbarrier  # turnstile
for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
    if cta == 1:
        for j in seq(0, lag):
            Arrive(cuda_in_order, 1) >> tb[cta, j] >> tb[:, j]
    for k in seq(0, K):
        Arrive(cuda_in_order, 1) >> bar[cta, k] >> bar[:, k]
        if cta == 0:
            Arrive(cuda_in_order, 1) >> tb[cta, k] >> tb[:, k]
        else:  # CTA 1 at iteration k waits for CTA 0 to reach iteration k + lag
            Arrive(cuda_in_order, 1) >> tb[cta, k + lag] >> tb[:, k + lag]
            Await(tb[cta, k + lag], cuda_in_order, 0)
        Await(bar[cta, k], cuda_in_order, 0)
    if cta == 0:
        for j in seq(K, K + lag):
            Arrive(cuda_in_order, 1) >> tb[cta, j] >> tb[:, j]
```

Both `lag=1` and `lag=2` pass sync_check, but `lag=2` deadlocks in CUDA:
* CTA 1, in iteration 0, waits for CTA 0's `tb` arrive from iteration 2.
* CTA 0 is blocked at `Await(bar[0,1])` waiting for CTA 1's iteration-1 `bar` arrive.

The trace runs all of CTA 0 first. Each `Await(bar[0,k])` sees only CTA 0's own arrive on `bar[0,k]`, and CTA 1's turnstile Awaits find everything already arrived.
Without a `seq` loop, a turnstile inside a single loop body can't deadlock, because CTA 1's Arrive precedes its turnstile wait. You need a lag of at least 2 iterations.

Under literal counts with a strict Await (`count == E`), CTA 0's first `Await(bar[0,0])` sees 1 of 2 arrives and fails with "No forward progress".
`lag=1` is rejected too: it's valid CUDA, but the serialized trace can't show it. That's the same conservatism as the non-multicast case.

Other experiments (clusterDim=4, `Arrive >> bar[m,n,k] >> bar[m,:,k]`, convergence check off):

| Program | Today | Convergence check off |
|---|---|---|
| `seq k` loop inside `n_cta` loop; Awaits in a later loop nest (valid CUDA) | rejected statically | PASS |
| Same, with `if n_cta == 0:` around Arrive (invalid) | rejected statically | FAIL: "No forward progress" |
| `test_arrive_incompatible_multicasts_positive` body (2 arrives, ring depth 1) | static checks pass (`cuda_cpu_test` never runs sync_check) | FAIL: "Arrived twice" |
| `Arrive >> bar[m,n,0] >> bar[m,:,0]; Await(bar[m,n,0])` in the same CTA body (valid CUDA) | PASS (weak edges) | PASS |

The third row is invalid in CUDA too (its second Await reuses parity 0), even though the test is labeled "positive".

## Literal-count design sketch

* `BarrierEnvAlloc` carries `E`, computed in Python from `SyncInfo.multicasts` and the extents. It must equal codegen's `len(cta_xor_list)`.
  Count per Arrive instance (thread collective), or count per thread so that it literally equals the `mbarrier.init` count. Per-thread counting would allow relaxing `consistent_arrive_thread_count` later.
* `on_arrive`: for each referenced barrier `b`, record `PendingAwait{b, count_b}` and then increment `count_b`. Error if `count_b > E`.
  The home is kept for logs only.
* `on_await`: "No forward progress" unless `count == E`. `max_arrive_count = count - 1 - N` is unchanged.
* `free_barriers`: `count == E`, replacing "one-shot arrive".
* Pre-arrive (`sync_check.py:426-430`, one arrive per element today): set the count to `E`.
* Trailing barriers (`syncv_table.cpp:794`): the `count == 0` check must loosen to `count < E`.
  Otherwise the multicast TMA trailing exprs (`raw[cta_m, :, k]`, `tk_gemm_util.py:1729`) fail in any schedule where a peer's Arrive precedes this CTA's TMA in the trace.
  The current gemm is fissioned (all TMAs before all arrives), so it would still pass.
  Real correctness comes from the `expect_tx`/`complete_tx` byte accounting, which Exo does not model.
* `home_barrier_expr()` stays as the "own shard" expression for distributed memory analysis and codegen (rename?). It drops out of the semantics.
* With strict Await, the last row of the table above becomes a "No forward progress" error. Programs like it need the Awaits fissioned into a later loop nest.
  Check `tests/cuda/test_sync_error_Sm90a_gemm.py` (multicast, ncta_M=2, runs on CPU) and the sporkbench Sm90 examples for this.

Not audited: ring-buffer slot reuse for barriers (the barrier `SyncEnvAccess` with `access_multicasts`). It is orthogonal to convergence.

## Recommendation

1. Implement literal counts with a strict Await in camspork and `sync_check.py`.
   Then delete the convergence check (`distributed_memory.py:599-623`) and rewrite `BarrierMulticast.tex`, `ArriveSemantics.tex`, `ArriveAwaitPairing.tex`, `gHomeBarrier.tex` and `MbarrierUsage.tex` (one-shot → expected count).
   Do **not** delete the convergence check before literal counts land (see counterexample).
2. Tests: the turnstile program (`lag=1,2` rejected), the experiment table rows, and fixing `test_arrive_incompatible_multicasts_positive`.
3. Remove `N` from Arrive (see (b)).
