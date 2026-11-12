# Plan: camspork Trace-based Testing (evaluation of the sketch)

**AI Disclosure: written by Claude Opus 5.5, not yet reviewed by David Zhao Akeley** (except where a line says "David Zhao Akeley:").
Claims are Claude's reading of the code unless marked **[tested]** (exercised by the prototype tests).
See "Unverified claims" at the end for the ones most worth checking.

Evaluates [plan_sketch_camspork_test.md](plan_sketch_camspork_test.md) (human sketch) against the code, 2026-11-07.
Related: [plan_multicast_barrier_counts.md](plan_multicast_barrier_counts.md) (mbarrier counts may change; the reference keeps all mbarrier behavior in one class),
[plan_remove_vis_flags.md](plan_remove_vis_flags.md) (current qual-tl model).

**Verdict: the sketch works.** A prototype on exo branch `akeley98/camspork_trace_proto` runs an Exo proc through `sync_check`,
traces probed array elements from camspork's no-log path, and compares the trace against a hand-written Python reference as sets.
Both prototype tests pass (positive, plus a negative that catches a wrong `CudaWarps` range).

## Prototype (exo `akeley98/camspork_trace_proto`, uncommitted)

| File | Change |
|---|---|
| `camspork/lib/syncv/syncv_table.{hpp,cpp}` | Debug Inspection section only (~45 lines): `debug_get_assignment_record_data`, `debug_get_barrier_data`, and `VisRecordDebugData::flags`. No core code touched. |
| `camspork/lib/program/exec.{hpp,cpp}` | `ProgramEnv::{trace_filename, trace_probes}`, C API `camspork_set_trace_filename` / `camspork_add_trace_probe`. `ProgramExec::trace_sync` writes one JSON line before and after each Fence/Arrive/Await/JoinThreads. |
| `camspork/camspork.py` | `ProgramEnv.set_trace(filename, [(var, idx)])` |
| `spork/sync_check.py`, `API.py` | `p.sync_check(_camspork_trace=CamsporkTraceRequest(filename, [x[0, 0]]), **sizes)` (probes are `StrElemName`s); sync_check fills in `request.stmt_srcinfo`. Probes name variables by `str(Sym)` and assert it is unambiguous. Installed only on the first (unfiltered) run, not on the single-sync-var rerun. |
| `spork/camspork_ref.py` | Python reference: `SyncEnvRef`, `Mbarriers` (all mbarrier logic), trace parse/compare/dump. |
| `tests/cuda/test_camspork_trace.py` | `test_named_numeric_warps_{positive,negative}`: `CudaWarps(1, 3, name="B")` writes, CTA Fence, `CudaWarps(name="A")` writes, Fence. |

How it fits the sketch's constraints:

* **No-log path is what's tested.** `trace_sync` lives in `ProgramExec<AllowLog>` and does not depend on `AllowLog`.
  `syncv_table` is still called with the `nullptr` logger (the `SyncvTrivialLogger` instantiation); the trace only reads state through the debug interface afterwards.
  Nothing uses `SinglePositionFilter`, so hash tables and assignment-record copy-on-write run as in production.
* **Read-out needed almost nothing new.** `debug_get_vis_record_data` and `debug_get_read_vis_record_ids` already existed.
  The prototype adds the mutate list, barrier arrive/await counts, and the VisRecord `flags`.
* **Not allocated / out of range:** the probe prints `"mutate": null, "read": null` (and omits barrier counts). The reference uses `None` for a freed or not-yet-allocated variable.
  "Any live allocation of that name" works because camspork has one var slot per name.
* **Formats (excut-style round trip; David Zhao Akeley chose this, 2026-11-07):** both sides write the same JSONL schema to the pytest `tmp_path`, one line per sync event phase:
  `camspork_trace.jsonl` (exec.cpp) and `reference_trace.jsonl` (`SyncEnvRef.write_trace`).
  Both are read back by the same `parse_trace`, then compared. The same printer writes `camspork_trace.txt` and `reference_trace.txt`, which diff line by line (only the `@` source lines differ when traces match).
  * Schema: `{phase, stmt, stmt_id, src, threads: [[lo, hi)...], cuboid, probes: [{var, idx, mutate, read, arrive_count?, await_count?}]}`.
    Records are `{ob: [[lo, hi, qual_bits]...], aw: [["bar[i]", n]...], mask, flags}`.
    The only flag left is `vis_record_before_alloc_flag` (managed ring buffer alloc records), which the reference ignores.
  * Source lines on both sides: camspork has `stmt_id`, which `parse_trace` maps to the Exo proc line via `get_stmt_srcinfo`; reference entries carry the `mkref` Python line (`get_srcinfo`, like `xrg(...)`; helpers pass `depth=`).
    A mismatch message prints both, plus the JSONL file:line of each entry.
  * Deliberately not copied from excut: variable deduction (e.g. for task IDs) and filtering unmentioned events. Camspork is predictable enough that these add risk without benefit (David Zhao Akeley).
* **Comparison:** per probe and kind (mutate/read), compare **sets** of `TraceVisRecord(ob, aw, mask)`.
  (`camspork_ref.py`'s `VisRecord.freed` and its `flags & 2` decoding are dead and can be removed.)
  Thread sets are interval tuples (the CPU is `[0, UINT32_MAX)`, so nothing can be enumerated).
  The sync statement's own thread cuboid is compared too, which checks how the *sync* statement was lowered, not just the accesses.
* **Thread IDs are hand-computed** in the reference (like excut `mkref_*`), so CollTiling → `ThreadsFor` lowering is checked independently.
  Qual-tl bits, masks and sync-tl bits come from Exo's tables (`InstrQuals`, `make_qual_tl_mask`, `Sync_tl.get_timeline_set_bits`); those are configuration, not under test.

## Findings, per sketch bullet

### Reference sync env (`SyncEnvConcat.tex`)

Straightforward. The prototype implements Post/Pre access (convergent and separate-tid), Fence, Arrive and Await as written, using `syncsWith` with the `Tid × QtlMask` mask.
`Aw` is stored as `{barrier: min arrive count}`, which stands for the infinite set `{PendingAwait(b, z) | z ≥ n}`; camspork's `PendingAwaitRay` means the same thing (see the comment in `syncv_table.hpp`).

### `SyncEnvImpl.tex` optimizations: what actually exists

1. Mutate discards prior reads / non-atomic mutate discards prior mutates (`copy_on_write_update` in `checked_on_access_impl`). **Both removed** from camspork:
   * prior mutates: exo `03434218` (finding below);
   * prior reads: exo `4cb8da49`, unsound across tasks (see "Finding: ... unsound across tasks").

   The reference uses spec semantics: all prior records are kept.
   **Finding (2026-11-07): the mutate-clears-mutates rule is load-bearing, not just a performance optimization.**
   * Experiment: exo `akeley98/experiment_keep_prior_mutates` (`1527e110`), env var `CAMSPORK_KEEP_PRIOR_MUTATES`.
   * Selected sporkbench kernels (pldi Sm80 starter, Sm90a f32/bf16 GEMMs, tk_fa, unflash_attn; 56 sync_check calls) all still pass. Total time is 1.06x, worst case 2.1x on `unflash_attn_S_kernel`.
   * But `tests/cuda/test_sync_error_Sm90a_gemm.py::test_ping_pong_positive` fails with `RAW HAZARD @ A_smem[0, ...]`, and `test_ping_pong_wrong_wgmma_cg` now reports RAW instead of `free_missing_arrive`.
   * Cause: the managed ring buffer's alloc record (`set_managed_ring_buffer_barriers`; a mutate record with empty Ob and Aw on `war[i]`) is augmented by the producer's `Await(war, cuda_async_proxy_retired)`.
     The TMA write used to delete it. Now it survives, and the consumer, ordered only through `raw`, never syncs with it.
   * David Zhao Akeley: the alloc record must stay a mutate. It approximates "what happened to that memory before" (an older, aliased allocation of the recycled memory),
     so later reads and writes must be checked against it (RAW/WAW), not just the first write. Making it a read record would miss those bugs.
     The old rule was wrong reasoning that hid a real bug.
   * Fix (David Zhao Akeley): the producer's "data ready" Arrive on `raw` uses `cuda_async_proxy_retired` instead of `cuda_mbarrier_only`.
     The alloc record then syncs with that Arrive, picks up the `raw` pending await, and the consumer's `Await(raw)` covers it.
     Applied to the 4 Arrives in `platforms/Sm90/tk_gemm_util.py`, together with removing the special case, on exo `akeley98/endgame`: fix `aaaf93c3`, special-case removal `03434218` (2026-11-07).
     Codegen unchanged: only the printed `// Arrive(...)` comments in the Sm90a GEMM goldens and the remarks golden change. `tests/cuda` passes.
   * The `cuda_mbarrier_only` / `cuda_async_proxy_retired` docstrings in `spork/timelines.py` were updated by David Zhao Akeley (in `aaaf93c3`).
2. "Expand async observers to warp/warpgroup/CTA": this is the `gOooOpt` `thread_access_granularity`, applied **at access time** to non-convergent accesses with an out-of-order initial qual-tl,
   with TAG from `CollAnalysis.get_qual_tl_thread_alignment`. It is not applied at sync time as the bullet's wording suggests (doc imprecision; noted below).
   Reference plan: each test passes a literal TAG, per the sketch. Not in the prototype.
3. "Expansion at end of device tasks / launches" is `JoinThreads`, which **is not an amspork3 trace event at all**.
   Rule: any record with an observer among the joining threads gets `joining threads × (union of those observers' qual-tls) ∩ mask`.
   The reference implements this as a DEVIATION (prototype: `SyncEnvRef.join_threads`).

### Managed ring buffer alloc / free

Definitions: `amspork3/SmemAppendix.tex` (`allocTrDetail` / `freeTrDetail`).
camspork implements both in `exec.cpp` `SyncEnvManageRingBuffer`, for element `x[..i..]` with guard multicast group `g_b` and depth `d`
(exo `5f59ad7c`; it replaced camspork's `free_on_arrive` mechanism, whose eager clearing of records would have needed a reference deviation):

* **Alloc** (`is_free=False`, `set_managed_ring_buffer_barriers`): a mutate record with empty Ob and,
  for each guard, `PendingAwaitRay(g_b, z_eac - 1)`, where `g_b` is from `g[..i..]` for data, or `g[..i-d..]` for a self-guarded barrier ring (omitted when `i < d`).
  It carries `vis_record_before_alloc_flag` until first augmented, so a failed check against it reports `use_before_alloc`.
* **Free** (`is_free=True`, `on_check_pending_awaits`): every mutate and read record must contain `PendingAwait(g_b[..i+d..], z_eac - 1)` for **each** guard,
  else `free_missing_arrive`. Vacuous when `i+d` is out of range for a self-guarded ring.

This matches the spec directly, so the reference needs no deviation.
Reference and tests: see "Managed ring buffers + mbarriers" below.

* **Non-ring SMEM:** `allocTrDetail` = (). `freeTrDetail` = `TrPreAccess(mutate, x, z, {tid}, Q_pre(mem), ∅)` for every tid of the current box (non-convergent).
  sync_check refuses unless the box equals the cluster (over-approximation for SMEM reuse across the cluster).
  The mutate/read records are then dropped (alpha-renaming in amspork3; `clear_visibility` in camspork).

### Other deviations from amspork3 found (excluding task-ID linearization, one-shot, forward progress)

1. **Multicast Arrive:** only the home barrier's count is read and incremented, and all targets get `PendingAwait(target, home count)`.
   The spec's `Tr-Arrive` uses each barrier's own `σ` count. Already covered by [plan_multicast_barrier_counts.md](plan_multicast_barrier_counts.md); the reference's `Mbarriers.arrive` is the single place to change.
2. **camspork bug: atomic observers are never masked.** The spec intersects `Tid × Q_atom` with `Tid × QtlMask` (`envConcatPost`).
   camspork adds `[0, UINT32_MAX) × atomic` unmasked (`alloc_vis_record`, whose mask `REQUIRE` explicitly exempts atomic bits), and its access checks never mask.
   Augment does mask, but L2 can never contain atomic bits, so that is moot.
   The real mistake is upstream: atomic qual-tls are never in any memory's `QtlMask` (checked: `make_qual_tl_mask() & atomic_bits == 0` for `CudaGmemLinear`, `CudaSmemLinear`, `CudaRmem`),
   because `make_qual_tl_mask` is deduced from `qual_tl_dict` and atomic qual-tls come from `AtomicityInfo` instead. camspork papered over this.
   David Zhao Akeley: the spec is correct; camspork is wrong. Probably drop the auto-deduction of `QtlMask` (some other time).
   Fixing camspork alone (masking atomics) would make existing atomic-after-atomic sync checks fail until the masks include atomic qual-tls.
   The reference raises `NotImplementedError` for atomics until then; when implemented it follows the spec (masked).
   **Status (2026-11-07, [plan_QtlMask_atomics.md](plan_QtlMask_atomics.md), exo `akeley98/camspork_trace_proto`): `QtlMask` half fixed.**
   `make_qual_tl_mask` is no longer deduced from `qual_tl_dict`; it is all qual-tls of the memory's `QualMaskClass`, so GMEM/SMEM masks now include both atomic qual-tls.
   The atomic exemption in the `alloc_vis_record` mask `REQUIRE` is removed. `tests/cuda` passes.
   Still open: camspork's atomic observers (`[0, UINT32_MAX) × atomic`) and access checks remain unmasked; now that the masks contain the atomic bits, masking them should be a no-op for data RAM.
3. **`JoinThreads`** (above): a camspork-only sync event. David Zhao Akeley: JoinThreads will not be in the spec; the reference keeps it as a documented DEVIATION.
4. **Implicit syncs from `sync_check.py` lowering** show up in every trace: before the device function, `Fence(cpu ∪ cuda_stream, cuda_stream)` by the CPU;
   `JoinThreads` at the end of each device task; `JoinThreads` then `Fence(cuda_stream, cuda_stream)` by the CPU after the function.
   This is Exo lowering, not the spec. The reference keeps it in one helper (`ref_device_function` in the prototype test; move it into `camspork_ref.py`).
5. **Free check vocabulary:** `free_missing_sync` (non-ring SMEM), `free_missing_arrive` (managed ring buffer) and `use_before_alloc` are hazard *names* only; they don't change what is legal.

### Task IDs ("paper over")

* camspork tid = `task_index × domain_num_threads + local tid`, and `task_index` is **lazy**: it increments only when a cuboid is used after a `TasksFor` iteration.
* So `task_index` is the lexicographic position of the task in the `cuda_tasks` loop nest:
  the nest must be perfect (`loop_modes.py` `validate_loop`), and `sync_check.py` ends every device task with `JoinThreads`, which uses the cuboid, so each task bumps it exactly once.
* Reference (2026-11-11): `SyncEnvRef(..., task_shape=...)`; `task=` arguments take an index tuple (or an int for a 1-D nest), linearized by `task_linear_index`.
  Tested by `test_tasks_2d` (2 x 3 nest) and `test_tasks_2d_transposed_negative` (a column-major reference disagrees on thread sets).

### Fixed-width domain

* `ThreadCuboid::max_dim = 8` and tids are `uint32_t`.
* The reference has neither limit: interval thread sets, and Python cuboid expansion of any dimension.
* Tests should still stay under 8 dimensions because camspork itself `REQUIRE`s it.

### Remove camspork excut (not done in prototype)

Footprint to delete:

* `program/camspork_excut.{hpp,cpp}`
* `SyncvLogRequest::p_excut_actions`
* in `syncv_table.cpp`: `SyncvRealLogger::excut_*` (~150 lines), the `SyncvTrivialLogger` stubs, and 6 call sites in `checked_on_access_impl` / `set_managed_ring_buffer_barriers`
* `ExcutVisRecordFreeOnArrive` (already dead: never constructed)
* in `exec.cpp`: `ProgramExecLogBase` (file, `flush_excut_log`), and `log_barrier_helper`'s excut branch
* the C API `p_excut_filename` arguments, and `ProgramEnv.exec(excut_filename=)` in `camspork.py`

After that, `ProgramExec<true>` exists only for the history log.

## Risks / gaps found by prototyping

* Accesses are not traced, only sync events. A wrong access followed by an overwrite with no sync in between is invisible.
  David Zhao Akeley: accepted as a blind spot. The primary goal for now is to observe that the initial access threads are correct and that syncs grow vis records as expected,
  so tests avoid the blind spot by construction: never create an access that is not followed by a sync.
* Probe names are `str(Sym)`; a probed name shared by several Syms (e.g. after inlining or unrolling) asserts out (unprobed duplicates are fine since 2026-11-11). Could take `Sym`s directly.
* Non-convergent accesses split into per-thread records in the reference. That's fine for test sizes (guarded at 2^20 threads).
* The reference reuses Exo's qual-tl tables. A wrong table entry would show up identically on both sides (out of scope by design).

## Decisions (David Zhao Akeley, 2026-11-07)

* **Negative tests** ignore the trace entirely: they succeed as long as both camspork (syncv) and the reference fail, with no further inspection.
  The reference does not need to predict which hazard, where.
* **Initial scope** avoids mbarriers entirely (Arrive/Await are tested first through commit groups and cluster sync), managed ring buffers, atomics, and multi-dimensional tasks.
  So the ring buffer `allocTrDetail` / `freeTrDetail`, multicast arrive counts, and the `QtlMask` atomics fix are not needed yet.
* **camspork excut removal:** deferred.
* (2026-11-11) mbarriers and managed ring buffers are now in scope; see "Managed ring buffers + mbarriers" below.

## Partial implementation (2026-11-07, exo `akeley98/camspork_trace_proto`)

Scope per the decisions above. `tests/cuda/test_camspork_trace.py`: 10 tests, all passing.

* **Passing (8):**
  * named + numeric `CudaWarps`, positive and a negative trace mismatch;
  * strided `cuda_threads(unit=2 * cuda_warp)`;
  * `cp.async` (out-of-order, TAG 1) with `Sm80_CommitGroup` Arrive/Await;
  * windows: big write then small reads, small writes then big read, partially overlapping;
  * missing-Fence negative (both reject).
* **Also passing:** `test_gmem_join_threads` and `test_cluster_sync`.
  They were strict xfails for the Q_stream finding below until Q_stream was defined as `cuda_stream_sync`.

Reference (`spork/camspork_ref.py`):
* sync statements record their own trace entries;
* `Barriers` holds all barrier counting and the one-shot checks; the mbarrier multicast rule is isolated in one unused method;
* TAG grouping for non-convergent accesses; window accesses;
* SMEM free check (`freeTrDetail` for non-ring SMEM);
* `cuda_device_function` per amspork3 `C-CudaDeviceFunction` (Q_stream = `cuda_stream_sync`'s qual-tl), plus `device_task`;
* prior mutates kept.

Helpers: `access_quals`, `check_trace` (also flags vacuous probes), `check_both_fail`.

Findings:
* **Spec vs lowering (C-CudaDeviceFunction):**
  * `sync_check.py` uses `cuda_stream_sync` bits for both implicit Fences.
  * The spec's set was then Q_cuda, read as all qual-tl except `cpu_in_order_qual`. Compared with it, `cuda_stream_sync` lacks `wgmma_rmem_fenced_qual` and `tcgen05_smem_qual`.
    `tcgen05_smem_qual` is in the SMEM/GMEM masks, so GMEM records alive at the device function's final Fence differed.
    That reading also included the two atomic qual-tls, which camspork `REQUIRE`s never appear in a sync-tl.
  * David Zhao Akeley: the set should be whatever `cuda_stream_sync` is, renamed Q_stream (`\sporkQtlStream` in amspork3). The reference uses that; `ControlSemantics.tex` still has "TODO define Q_stream".
  * The spec's Fences are by all threads; camspork uses the "CPU" cuboid `[0, UINT32_MAX)`, which covers every linearized tid, so that part agrees.
* **Barrier identity is per allocation**, matching the spec's alpha-renaming per loop iteration and camspork's distinct barrier ids. The first reference keyed barriers by name and wrongly synchronized one thread's Await with another thread's `cg`; the trace caught it.
* **Scalar probes:** camspork scalar var slots always look allocated (no "unallocated" state). A scalar with no AssignmentRecord prints as `null` on both sides; a scalar barrier with id 0 is not live.
* **Trace plumbing:** barrier names in pending awaits are recorded at `BarrierEnvAlloc`, since PendingAwaits outlive the barrier's var slot.

## Typed reference (2026-11-11)

`camspork_ref.py` no longer identifies anything by raw strings or dicts (excut style: slotted dataclasses, frozen where hashed; enums in code, strings only in JSON):
* Names: `StrVarName` (x, `str(Sym)`), `ElemIdx` (z*), `StrElemName` (x[z*]; probes and traced barriers), `AllocVarName` (x#n, the n-th allocation of x, counted per x), `AllocElemName` (x#n[z*]; keys the reference's sync env and barrier state, projected to `StrElemName` in traces).
* Enums: `AccessKind` (mutate/read), `SyncStmtKind`, `TracePhase`.
* Records: `QualObservers`, `PendingAwaitRay` / `TracePendingAwaitRay`, `VisRecord` / `TraceVisRecord`, `ElemSyncEnv`, `AccessQuals`, `BarrierState`, `BarrierCounts`, `AllocRing`.
* Traces: `TraceEntry` / `TraceProbe`; the module docstring documents the JSONL schema with types (`TraceEntryJson`, `TraceProbeJson`, `VisRecordJson`).

## Managed ring buffers + mbarriers (2026-11-11, exo `akeley98/amspork3_SmemAppendix`)

Reference (`spork/camspork_ref.py`), per `SmemAppendix.tex`:
* `alloc_mbarrier`: a self-guarded mbarrier ring with its own sync env (QtlMask = `{cuda_mbarrier_qual}`).
  `arrive` / `await_` first do the read access `sync_check.py` emits: `cuda_mbarrier_qual`, non-convergent on every multicast target (Arrive), convergent on the home barrier (Await).
  This access is `sync_check.py` lowering; the spec only states the obligation ("Any Arrive or Await of x_b[z_b] must happen after awaiting for the previous phase").
* `alloc(..., ring=RingGuard(guard, ring_dim, ring_depth, guard_multicast))`: SMEM data ring guarded by an mbarrier array.
  `guard_multicast` is hand-written per test (like thread IDs) in the LoopIR `SyncStmt.multicasts()` format (one flag tuple per barrier expression),
  and gives `guards()`: the union over expressions, as camspork's `eval_multicast` does (not a product over all multicast dimensions).
* `free` / `free_barrier`: the ring free check (`freeTrDetail`) runs before the arrive count checks, as in `sync_check.py`.
* `alloc_barrier` now rejects `per_thread` (mbarriers must use `alloc_mbarrier`); the existing mbarrier tests were switched over and still pass.
* `VisRecord.freed` is removed; the reference ignores the trace's `flags`.

Tests (`tests/cuda/test_camspork_trace.py`, "Managed ring buffers" section; 30 tests in the file pass):
* `test_mbarrier_ring`: 4 phases on 2 physical mbarriers, probing every phase; Arrives by warps 0-1, Awaits by warp 0 and warp 1 separately.
* `test_mbarrier_ring_free_negative`: only warp 0 arrives, so warp 1's Await record on phase k never gets phase k + 2's PendingAwait (`free_missing_arrive` on both sides).
* `test_mbarrier_ring_missing_await_negative`: Arrive on phase 2 before any Await of phase 0 (camspork `use_before_alloc`; reference hazard).
* `test_smem_ring`: producer/consumer `war` / `raw` ring as in the Sm90 GEMMs (`war` has K + D phases, first D pre-arrived), probing data and both barriers.
* `test_smem_ring_early_war_arrive_negative`: the consumer releases the slot before reading it (`free_missing_arrive` on both sides; nothing else fails).
* `test_smem_ring_multicast_2cta`: `war` and `raw` multicast across a 2-CTA cluster. The trace shows the alloc record's `Aw` on both guards (`gAwait` multicast expansion).
* `test_smem_ring_cross_4cta`: the GEMMs' cross-shaped multicast (`>> war[m, n, k] >> war[m, :, k] >> war[:, n, k]`) on a 2 x 2 cluster; `Aw` on 3 guards, not 4.

Checked by mutating the reference (each caught by at least one test): no `guards()` multicast expansion; product instead of union over expressions; `z_eac` instead of `z_eac - 1`; free check skipped; no mbarrier read access.

Findings:
* **Blind spot (David Zhao Akeley): `gNext`'s multicast expansion is untestable.** An Arrive always covers the whole multicast group, so either all of `gNext` is missing or none of it.
  Fixable only with arbitrary distributed shared memory access, which is out of scope.
* **Multicast WAR needs a multicast RAW.** A first version of the 2-CTA test had `raw` CTA-local. Both sides rejected it:
  CTA 0's consumer arrived on CTA 1's next `war` phase with no ordering after CTA 1's producer awaited the current phase (a real race).
  The GEMMs multicast `raw` too. Producers and consumers also need separate CTA loops, or the serialized trace has no forward progress (the documented conservatism).
* **Hazard names differ** for the same rejection (e.g. camspork `use_before_alloc` vs the reference's generic hazard); negative tests don't compare them, per the decisions above.

## Finding: "a mutate discards prior reads" is unsound across tasks (2026-11-07)

David Zhao Akeley's counterexample:
* Each task does a generic read of `g[0, 0]`, then `Fence(cuda_in_order, cuda_generic_and_async_proxy)`, then a TMA reduce (atomic) into `g`.
* There is no hazard within a task.
* With 2 tasks, task 1's reduce races task 0's read (WAR). camspork misses it because task 0's own reduce discarded task 0's read record.

Tests (`tests/cuda/test_camspork_trace.py`, `test_read_then_reduce_*`):
* **1 task:** a full trace comparison. It passes, and it is the first test of atomics: TMA reduce with `Tid x Q_atom` masked, TAG 32. This needs exo `7c82d4a0` (QtlMask includes atomic qual-tls).
* **2 tasks, every task reads:** both camspork and the reference reject. But the rejection comes from task 1's *read* (RAW vs task 0's reduce), which masks the WAR.
* **2 tasks, only task 0 reads:** the unsound case. Before the fix, camspork accepted it (strict xfail).
  The spec-semantics reference rejects at `g[0, 0]`, thread 32 (task 1), during task 1's reduce.
* Experiment, not committed: with camspork's read clearing disabled, the same case reports `WAR HAZARD`, and the 1-task case still passes.

**Fixed** in exo `4cb8da49` (on `akeley98/camspork_trace_proto`):
* camspork keeps prior read records on a mutate;
* the reference's `discard_reads_on_mutate` option is removed;
* `test_read_then_reduce_2_tasks_negative` passes;
* `tests/cuda`: 370 passed.

Cost of dropping the rule (exo `akeley98/experiment_keep_prior_reads`, `e3425bdf`, env var `CAMSPORK_KEEP_PRIOR_READS`):
* sporkbench, 19 procs (17 with managed ring buffers, incl. ping-pong and split-K), 76 calls:
  * total sync_check time 1.02x; ring-buffer procs 1.02x;
  * worst case `unflash_attn_S_kernel` 1.18x (0.225 -> 0.264 s);
  * no sync_check failures in either mode.
* `tests/cuda`: only `test_camspork_trace` changes.
  * `test_window_overlapping` and `test_read_then_reduce_1_task` keep read records, and match the reference with `discard_reads_on_mutate=False`.
  * The 2-task strict xfail now passes.

## Proposed order

1. Move `ref_device_function`, plus task/cluster helpers, into `camspork_ref.py`; add the free-trace hook. (srcinfo: done.)
2. ~~Managed ring buffers + mbarriers in the reference~~ (done 2026-11-11). Still open: port a reduced `test_3cycle_mbarrier` (name restriction + numeric range under warpgroup loop).
3. Non-convergent out-of-order accesses with literal TAG (cp.async on Sm80), atomics after the `QtlMask` fix, strided collectives (`cuda_threads` with `unit=2 * cuda_warp`, as in `test_simple_named_warps`).
4. Copy-on-write tests: big window then small windows (de-convergence), the reverse (no false merging), partially overlapping windows.
5. Delete camspork excut.
6. Then tests for [plan_multicast_barrier_counts.md](plan_multicast_barrier_counts.md) changes.

## Doc notes (spork/docs, not caused by the rewrite)

* `amspork3/SyncEnvImpl.tex` bullet 2 ("the same for prior mutates after tracing non-atomic mutates") no longer matches camspork after exo `03434218`.
* `amspork3/SyncEnvImpl.tex` bullet 3 describes the TAG expansion as tied to the *synchronization's* scope; the implementation applies it at access time from per-qual-tl Fence/Arrive alignment, only for non-convergent out-of-order accesses.
* `amspork3/SyncEnvImpl.tex` bullet 4 (JoinThreads expansion) describes an event absent from the trace grammar (`Traces.tex`).

## Unverified claims (Claude's reading only; worth a human check)

* **[tested]** only: the trace plumbing (both JSONL writers and the shared parser); the CTA Fence expansion; thread mapping of `CudaWarps(1, 3, name="B")` and `CudaWarps(name="A")`; the implicit CPU Fence / JoinThreads events (their thread sets only, with no records live at those points).
* Not tested, from reading code:
  * the multicast Arrive and JoinThreads descriptions;
  * the lazy `task_index` behavior;
  * the excut removal footprint;
  * the doc notes above.
* (2026-11-07) JoinThreads, Arrive and Await now run with live records and barrier-count probes (commit group, cluster sync), and match camspork apart from the Q_stream finding (resolved).
* (2026-11-11) **[tested]** camspork's managed ring buffer alloc records (data and mbarrier) match `SmemAppendix.tex`; the free check agrees on pass/fail for the cases above (not `gNext` multicast).
