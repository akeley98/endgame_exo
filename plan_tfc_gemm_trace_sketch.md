# TFC GEMM Trace

Throwaway special order for Thea Fields Collin to try to show the overall data movement pattern
of Sm90a ping-pong GEMM, Sm90a ping-pong GEMM with TMA multicast, Sm80 CUTLASS-y GEMM.

There will be an offline C++ trace generator that relies on Exo's camspork C++ only,
plus a Claude-generated web artifact that embeds the traces and shows them graphically
along with the pseudocode of the GEMMs.

This is primarily to show the data movement and thread assignment pattern,
only secondarily do we care about the synchronization.
The synchronization checking is mainly to keep me honest,
but also interesting to opt-into viewing in case a discussion starts there.


# Deviations from Exo-GPU / Traditional Parallelism

I'm presenting an idealized GEMM not what Exo-GPU actually implements.
Very important not to let anchoring bias override the following:

* I care only about the synchronization environment semantics,
  not my stepping-stone trace semantics and certainly not
  any nondeterministic "interleavings".
  Multiple Claudes have anchoring bias for traditional parallel
  semantics and that is not what I am implementing.

* `task_index` for observers will not be used (set to 0 always).
  The examples will be written from the perspective of a single cluster
  that computes GEMM tiles assigned by an opaque function
  `next_task() -> m, n`.
  I regret the opacity of the Exo `cuda_tasks` loop.

* Atomics and GMEM synchronization environment are not modelled for this demo.

* Managed ring buffers don't exist.
  SMEM is indexed by CTA (optional) and ring buffer slot (latter in range `[0, ring_depth - 1]`.
  Mbarriers are indexed by CTA (optional), ring buffer slot, then phase.

* Ping me if there's deviations below from my own semantics not mentioned here.


# Semantics Sketch

Requires info from the `amspork3` thesis to fully understand.
Pseudocode program statements types:

* Loops without loop mode: `for x in range(e_lo, e_hi):`

* If stmt: `if e:`

* (Not in Exo) `with blk_thr[e_cLo:e_cHi, e_tLo:e_tHi]:`
  For Thea specifically as an alternative to the complicated `cuda_threads` loop.
  `tidMap` of child stmts gives threads with CTA rank in range `[e_cLo, e_cHi)`
  and threadIdx in range `[e_tLo, e_tHi)`, if not overridden by a nested `blk_thr`.

* Assignment and reduction stmts: `w_1 := w_2`, `w_1 += w_2`;
  each window may carry window annotations,
  which are by-default hidden when viewing the pseudocode in the viewer artifact.
  Not in Exo: may have trailing barrier expr `>> w_bar`
  because I won't want the complication of instrs in the demo.
  TMA will look like `(w_1 := w_2) >> w_bar`.

* Sync statements similar to Exo;
  qtl parameters hidden by default in the viewer artifact.

* (Not in Exo) `x_m, x_n = next_task()`, see Symbolic Evaluation.
  Think of `x_m, x_n` as control-type variables (index) even
  though this assignment is not possible at all in Exo.

* (Not in Exo) `x = alloc_rmem[z^*]`, allocate RMEM (qtlConvergent = True).

* (Not in Exo) `x = alloc_smem[z^*]`, allocate SMEM data (qtlConvergent = False).
  Unlike Exo, there's no allocator imposing synchronization obligations.
  Technically wrong for the multicast example as we have to worry about
  `Cluster target block not present`.
  Just remind me if that exit sync is missing for that demo.

* (Not in Exo) `x = alloc_async_group[z^*]`, sync-exempt barriers.
  No solitary barrier requirement unlike real Exo (don't worry about this).

* (Not in Exo) `x = alloc_mbarriers[z^*, infinity](expected_arrive_count=z)`,
  allocate mbarriers (qtlConvergent = False) with given expected arrive count.
  As mentioned, my pseudocode will expect the `-2` dim to be the ring buffer index,
  `-1` dim to be the phase.
  This still has an allocator, see `Idealized Mbarrier Allocator`.

* (Not in Exo) `prepare_phase(x[z^*, z_phase])`, see `Idealized Mbarrier Allocator`.

* Pseudocode must be able to contain `# comments`.

Just hard-wire the example programs in a new C++ AST structure of
your choice and generate both the traces and the pseudocode from it;
don't generate some parser just for the throwaway demo.
The quality of sync-check errors doesn't have to be good
as that's not part of the demo, just to check my own pseudocode is correct.


## Symbolic Evaluation: Control Variables

Either some integer `z`,
or `tasks[z_t].m + z_off`
or `tasks[z_t].n + z_off`.
t-th call to `next_task` (0-indexed) gives `(tasks[t].m + 0, tasks[t].n + 0)`.


## Symbolic Evaluation: non-GMEM Data Array Elements

* `0`

* `A[tasks[z_t].m + z_off, z_k]`

* `B[z_k, tasks[z_t].n + z_off]`

* `A[tasks[z_t].m + z_off, z_k] * B[z_k, tasks[z_t].n + z_off]` (note shared value constraints)

* `\sum_{k=0}^{z_k} A[tasks[z_t].m + z_off, k] * B[k, tasks[z_t].n + z_off]`

Arithmetic giving not one of the above forms is an error at trace generation time.

Each array element is also associated with visibility records (`EnvVis`), read or mutate.

For pedagogical purposes, include separately and prominently the
thread(s) `(CTA rank, threadIdx)` that last assigned the value,
with no qtl etc. as the thread assignment is of greatest TFC interest for now.


## Symbolic Evaluation: GMEM

No visibility records and sync check, as mentioned.

`A` and `B` can only be read from;
give symbolic `A[...]` `B[...]` values.
`C[m, n]` can only be stored to;
error if stored value is not `\sum_{k=0}^{K} A[m, k] * B[k, n]`.

Earlier constraints imply `m, n` indices must have symbolic form
`tasks[z_t].m + z_off` and `tasks[z_t].n + z_off` respectively.


## Symbolic Evaluation: Barriers

Each barrier array element has an arrive count,
and for mbarriers only,
expected arrive count and visibility records.


## Idealized Mbarrier Allocator

For `x` of mbarrier arrays type only:

* Error to access `x[z^*, z_phase]` until after the phase is prepared
  with `prepare_phase(x[z^*, z_phase])`.

* If `z_phase > 0`, this adds to the synchronization environment
  `EnvEff(mutate, x, (z^* ++ z_phase)), EnvVis({}, Aw', Qtl_mask')`
  where `Aw' = {PendingAwait(x, z^* ++ (z_phase-1), z) \mid z >= expected_arrive_count-1}`
  and `Qtl_mask' = {cuda_mbarrier_qual}`.

* At program exit, `x` has a free check:
  for every prepared `x[z^*, z_phase]` with `z_phase > 0`,
  all visibility records (read or mutate) of `x[z^*, z_phase-1]`
  must contain `PendingAwait(x, z^*, expected_arrive_count - 1)`.

This really should be similar to the SMEM Appendix except with the
managed ring buffer dimension explicitly split to ring and phase dimensions,
and the number of phases being discovered at runtime rather than allocated upfront.
Ping me if not.


# Camspork Usage

Use only the code in the `syncv` directory and avoid the messy
`ProgramEnv` and such built on top of it.
This means that you have to embed `qtl` names inside the demo
and come up with your own qtl bit assignments; don't have to copy Exo exactly.

Use the `akeley98/tfc_gemm_trace` branch and just compile the new C++
trace generator code into `libcamspork.so` and generate it with
a little Python ctypes wrapper or something.
Avoid changes to the existing code other than adding public accessors if needed.

The reason for recycling this mess of C++ is at least it's tested and
I'm reasonably satisfied it actually implements the semantics
promulgated in the `amspork3` thesis besides the known documented
deviations. Avoid the "out-of-order non-convergent abstract machine optimization"
entirely by passing `thread_access_granularity = 1` always.
`JoinThreads` will never be used.

For this throwaway code I'll be satisfied with no testing,
just relying on the existing C++ code and me eyeballing the final artifact.

AFAIK, `blk_thr[e_cLo:e_cHi, e_tLo:e_tHi]` can be implemented with a `ThreadCuboid`
with `task_index = 0`, `domain = {CTA per cluster, threads per CTA}`,
`offset = {e_cLo, e_tLo}`, `box = {e_cHi-e_cLo, e_tHi-e_tLo}`;
omit CTA dimension if `clusterDim = 1`.

For the mbarrier `prepare_phase`/free checks,
`set_managed_ring_buffer_barriers` and `on_check_pending_awaits`
seem to do exactly what's needed.
(Double check) the former adds a single mutate `VisRecord` that contains
`PendingAwait(x, z^*, expected_await_count - 1)` for each `x[z^*]` in the barrier array;
the latter checks each mutate/read `VisRecord` contain *all* of
`PendingAwait(x, z^*, expected_await_count - 1)` for each `x[z^*]` in the barrier array.
AFAIK this demo needs only length-1 arrays as we don't have data SMEM managed ring buffers.


# Viewer Artifact

Each step of execution is uniquely identified by the executed stmt's
number and the in-scope loop iterator values.
The trace generator gives a snapshot of the program state at each step
indexed as above.
PLEASE save space by putting all VisRecord values in a separate de-duplicated table
shared by all snapshots and identify VisRecord values by ID elsewhere.

Show the pseudocode with statement numbers `S1`, `S2`... (I think S is conventional?)
Window annotations and sync stmt qtl parameters are collapsed to `...`
by default, hover over with cursor to see.

Have debugger like step forward and step backwards buttons
that select the currently-viewed snapshot.
Also, allow adjusting the (stmt id, iterator values)
that selects the snapshot directly with "select stmt id" widgets next to statement numbers
and adjust-iterator-value-up/down widgets next to loops.

Needs to be a status panel showing all variable and array values
(except GMEM) of the currently-viewed snapshot, with each array in tabular format.
Value shown is either `\bot` (not initialized) or a symbolic evaluation value.
For initialized data values, also show the threads
`(CTA rank, threadIdx)` that stored the value,
in compressed form `(z, z)..(z, z)`.

Click on an array element in the status panel to inspect the VisRecord(s)
of that array element in another panel.
Show observers in compressed form `(z, z)..(z, z) \times {qtl}`.
This is likely crazy verbose especially if qtlConvergent = False (not RMEM)
but it's whatever, not the main point of the demo (yet).

The C++ trace generator should separately be able to dump plain text
pseudocode of each demo, for me to attach separately.

See [plan_tfc_gemm_pseudocode_sketch.md](plan_tfc_gemm_pseudocode_sketch.md).
