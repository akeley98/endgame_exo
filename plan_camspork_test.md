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
| `spork/sync_check.py`, `API.py` | `p.sync_check(_camspork_trace=(filename, [("x", (0, 0))]), **sizes)`. Probes name variables by `str(Sym)` and assert it is unambiguous. Installed only on the first (unfiltered) run, not on the single-sync-var rerun. |
| `spork/camspork_ref.py` | Python reference: `SyncEnvRef`, `Mbarriers` (all mbarrier logic), trace parse/compare/dump. |
| `tests/cuda/test_camspork_trace.py` | `test_named_numeric_warps_{positive,negative}`: `CudaWarps(1, 3, name="B")` writes, CTA Fence, `CudaWarps(name="A")` writes, Fence. |

How it fits the sketch's constraints:

* **No-log path is what's tested.** `trace_sync` lives in `ProgramExec<AllowLog>` and does not depend on `AllowLog`.
  `syncv_table` is still called with the `nullptr` logger (the `SyncvTrivialLogger` instantiation); the trace only reads state through the debug interface afterwards.
  Nothing uses `SinglePositionFilter`, so hash tables and assignment-record copy-on-write run as in production.
* **Read-out needed almost nothing new.** `debug_get_vis_record_data` and `debug_get_read_vis_record_ids` already existed.
  The prototype adds the mutate list, barrier arrive/await counts, and the VisRecord `flags` (needed for "freed").
* **Not allocated / out of range:** the probe prints `"mutate": null, "read": null` (and omits barrier counts). The reference uses `None` for a freed or not-yet-allocated variable.
  "Any live allocation of that name" works because camspork has one var slot per name.
* **Formats:** camspork writes JSONL (`camspork_trace.jsonl`); the reference writes readable text (`reference_trace.txt`), both in the pytest `tmp_path`.
  Each line has `stmt_id` (`StmtRef.raw_data`). BONUS (not wired yet): `compare_traces(..., srcinfo=)` takes a callback; build `StmtRef(raw)` and call `get_stmt_srcinfo`.
* **Comparison:** per probe and kind (mutate/read), compare **sets** of `VisRecord(ob, aw, mask, freed)`.
  Thread sets are interval tuples (the CPU is `[0, UINT32_MAX)`, so nothing can be enumerated).
  The sync statement's own thread cuboid is compared too, which checks how the *sync* statement was lowered, not just the accesses.
* **Thread IDs are hand-computed** in the reference (like excut `mkref_*`), so CollTiling → `ThreadsFor` lowering is checked independently.
  Qual-tl bits, masks and sync-tl bits come from Exo's tables (`InstrQuals`, `make_qual_tl_mask`, `Sync_tl.get_timeline_set_bits`); those are configuration, not under test.

## Findings, per sketch bullet

### Reference sync env (`SyncEnvConcat.tex`)

Straightforward. The prototype implements Post/Pre access (convergent and separate-tid), Fence, Arrive and Await as written, using `syncsWith` with the `Tid × QtlMask` mask.
`Aw` is stored as `{barrier: min arrive count}`, which stands for the infinite set `{PendingAwait(b, z) | z ≥ n}`; camspork's `PendingAwait` means the same thing (see the comment in `syncv_table.hpp`).

### `SyncEnvImpl.tex` optimizations: what actually exists

1. Mutate discards prior reads; non-atomic mutate discards prior mutates: `copy_on_write_update` in `checked_on_access_impl`. Implemented in the reference (OPT).
2. "Expand async observers to warp/warpgroup/CTA": this is the `gOooOpt` `thread_access_granularity`, applied **at access time** to non-convergent accesses with an out-of-order initial qual-tl,
   with TAG from `CollAnalysis.get_qual_tl_thread_alignment`. It is not applied at sync time as the bullet's wording suggests (doc imprecision; noted below).
   Reference plan: each test passes a literal TAG, per the sketch. Not in the prototype.
3. "Expansion at end of device tasks / launches" is `JoinThreads`, which **is not an amspork3 trace event at all**.
   Rule: any record with an observer among the joining threads gets `joining threads × (union of those observers' qual-tls) ∩ mask`.
   The reference implements this as a DEVIATION (prototype: `SyncEnvRef.join_threads`).

### `free_on_arrive` (research answer)

The amspork3 grammar already has the hook: `TrPreAccess(..., Aw^free)`, and `syncsWith`'s `Aw` disjunct.
Proposed `allocTrDetail` / `freeTrDetail`, matching `exec.cpp` `SyncEnvManageRingBuffer` / `SyncEnvFreeManagedRingBuffer` / `SyncEnvFreeShard`:

* **Managed ring buffer, element `x[..i..]` with guard `g` and depth `d`:**
  * `allocTrDetail` = `TrPostAccess(mutate, x, z, Tid=∅, q_init=-, Q_atom=∅, mask(x), Aw^post = {PendingAwait(g[..i-d..], z) | z ≥ 0})`.
    It is omitted when `i < d` for a self-guarded ring. The `z ≥ 0` is camspork's `ForceArriveCount0`: any Await on `g[i-d]` satisfies it.
  * `freeTrDetail` = `TrPreAccess(mutate, x, z, Tid=∅, Q=∅, Aw^free = {PendingAwait(g[..i+d..], z) | z ∈ ℕ})`.
    This is vacuous when `i+d` is out of range. A record passes iff some Arrive including `g[i+d]` synced with it.
* **Non-ring SMEM:** `allocTrDetail` = (). `freeTrDetail` = `TrPreAccess(mutate, x, z, {tid}, Q_pre(mem), ∅)` for every tid of the current box (non-convergent).
  sync_check refuses unless the box equals the cluster (over-approximation for SMEM reuse across the cluster).
  The mutate/read records are then dropped (alpha-renaming in amspork3; `clear_visibility` in camspork).
* **The eager-clear deviation:** camspork's `ArriveUpdateCommand` empties a record (Ob and Aw) and sets `vis_record_freed_flag` as soon as an Arrive including `free_on_arrive` syncs with it.
  Any later access to that element is a `use_after_free HAZARD`, even one the spec would accept.
  Reference: keep spec semantics (the record keeps growing) and derive `freed(r) := r.aw ∋ g[i+d]`. Compare both sides with the record mapped to a `FREED` token (Ob/Aw ignored),
  and add one explicit "use after freeing Arrive is an error" rule as a documented DEVIATION.
  Edge case where the derived predicate and camspork differ: an access whose *trailing barrier* is `g[i+d]` (Aw at creation, never cleared by camspork). It is unlikely in real code; tests should avoid it or the reference can track "freed" as an event instead.

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
3. **`JoinThreads`** (above): a camspork-only sync event.
4. **Implicit syncs from `sync_check.py` lowering** show up in every trace: before the device function, `Fence(cpu ∪ cuda_stream, cuda_stream)` by the CPU;
   `JoinThreads` at the end of each device task; `JoinThreads` then `Fence(cuda_stream, cuda_stream)` by the CPU after the function.
   This is Exo lowering, not the spec. The reference keeps it in one helper (`ref_device_function` in the prototype test; move it into `camspork_ref.py`).
5. **Free check vocabulary:** `free_missing_sync` / `use_before_alloc` / `use_after_free` are hazard *names* only (flags on records); they don't change what is legal, except item "eager clear" above.

### Task IDs ("paper over")

* camspork tid = `task_index × domain_num_threads + local tid`, and `task_index` is **lazy**: it increments only when a cuboid is used after a `TasksFor` iteration.
* Prototype: the reference takes `threads_per_task` and a `task=` argument to `ref.tids`. That is exact for one task.
* Multi-task tests: canonicalize task indices by first appearance on both sides, or keep tests at one task per device function.

### Fixed-width domain

* `ThreadCuboid::max_dim = 8` and tids are `uint32_t`.
* The reference has neither limit: interval thread sets, and Python cuboid expansion of any dimension.
* Tests should still stay under 8 dimensions because camspork itself `REQUIRE`s it.

### Remove camspork excut (not done in prototype)

Footprint to delete:

* `program/camspork_excut.{hpp,cpp}`
* `SyncvLogRequest::p_excut_actions`
* in `syncv_table.cpp`: `SyncvRealLogger::excut_*` (~150 lines), the `SyncvTrivialLogger` stubs, and 6 call sites in `checked_on_access_impl` / `set_managed_ring_buffer_barriers`
* in `exec.cpp`: `ProgramExecLogBase` (file, `flush_excut_log`), and `log_barrier_helper`'s excut branch
* the C API `p_excut_filename` arguments, and `ProgramEnv.exec(excut_filename=)` in `camspork.py`

After that, `ProgramExec<true>` exists only for the history log.

## Risks / gaps found by prototyping

* Accesses are not traced, only sync events. A wrong access followed by an overwrite with no sync in between is invisible.
  Cheap fix: also trace before `DataFree` / `SyncEnvFreeShard` / `SyncEnvFreeManagedRingBuffer` (final state of the element).
* Probe names are `str(Sym)`; same-named Syms (inlined procs) assert out. Could take `Sym`s directly.
* Non-convergent accesses split into per-thread records in the reference. That's fine for test sizes (guarded at 2^20 threads).
* The reference reuses Exo's qual-tl tables. A wrong table entry would show up identically on both sides (out of scope by design).

## Proposed order

1. Move `ref_device_function`, plus task/cluster helpers, into `camspork_ref.py`; add the free-trace hook; wire `srcinfo` (BONUS).
2. Managed ring buffers + mbarriers in the reference (`allocTrDetail` / `freeTrDetail` above, `FREED` mapping); port a reduced `test_3cycle_mbarrier` (name restriction + numeric range under warpgroup loop).
3. Non-convergent out-of-order accesses with literal TAG (cp.async on Sm80), atomics after the `QtlMask` fix, strided collectives (`cuda_threads` with `unit=2 * cuda_warp`, as in `test_simple_named_warps`).
4. Copy-on-write tests: big window then small windows (de-convergence), the reverse (no false merging), partially overlapping windows.
5. Delete camspork excut.
6. Then tests for [plan_multicast_barrier_counts.md](plan_multicast_barrier_counts.md) changes.

## Doc notes (spork/docs, not caused by the rewrite)

* `amspork3/SyncEnvImpl.tex` bullet 3 describes the TAG expansion as tied to the *synchronization's* scope; the implementation applies it at access time from per-qual-tl Fence/Arrive alignment, only for non-convergent out-of-order accesses.
* `amspork3/SyncEnvImpl.tex` bullet 4 (JoinThreads expansion) describes an event absent from the trace grammar (`Traces.tex`).

## Unverified claims (Claude's reading only; worth a human check)

* **[tested]** only: the trace plumbing; the CTA Fence expansion; thread mapping of `CudaWarps(1, 3, name="B")` and `CudaWarps(name="A")`; the implicit CPU Fence / JoinThreads events (their thread sets only, with no records live at those points).
* Not tested, from reading code:
  * the `allocTrDetail` / `freeTrDetail` definitions;
  * that the `freed(r)` predicate matches camspork's eager clearing (and the trailing-barrier edge case);
  * the multicast Arrive and JoinThreads descriptions;
  * the lazy `task_index` behavior;
  * the excut removal footprint;
  * the doc notes above.
* The reference's JoinThreads, Arrive and Await rules ran only on empty environments; no test has a record live during them yet.
* `amspork3/Smem.tex` is empty; the `allocTrDetail` / `freeTrDetail` definitions above are candidates.
