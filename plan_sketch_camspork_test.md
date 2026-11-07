# Human Sketch of Wanted Camspork Trace-based Testing

* Look over amspork3 docs, most importantly `SyncEnvConcat.tex`,
  and implement a Python reference version of the documented synchronization environment and each trace event case.
  These will be used to generate expected outputs to compare against.

* Also implement reference versions of the optimizations mentioned in `SyncEnvImpl.tex`.
  All should be straightforward *except* for the old `spork_b/gOooOpt.tex` optimization.
  For this one, compute literal TAG values for test cases;
  don't rely on the untrusted Exo code.

* I don't know how to handle `free_on_arrive` yet; research this.
  It's useful that camspork3 eagerly clears visibility records for "deallocated" array elements,
  so error messages are located closer to the offending buggy code,
  but this is a major deviation.

* Be on the lookout for other deviations from the new amspork3 specs and tell me about them,
  except for the linearization of task IDs,
  and the not-yet-explained one-shot arrive/await stuff and forward progress stuff.
  Paper over the task ID somehow and just avoid these other features for now for the testing.
  I'm more interested that the core visibility record (envVis) state is correct
  and that parallel loops are correctly compiled to camspork.

* For SMEM allocation, in the Python reference,
  write actual implementations for the red allocTrDetail and freeTrDetail functions.
  So I have something to copy for the thesis that actually matches this whole mess.

* Avoid the fixed-width camspork domain limitation.

* Do the bare minimum edits to camspork's `syncv_table.cpp` to be able
  to read out visibility records and arrive counts.
  I am terrified of edits to this file.
  Also, I know the terminology is horrbly outdated (e.g. sync-tl = observer).
  Don't rely on the untrusted complicated history logging system for this.

* Do however get rid of all the old excut stuff in `syncv_table` and everything above that exposes it.
  This cruft never went anywhere.
  So the only "deeply integrated logging" left in `syncv_table` is the history logging.
  We will not be testing that as part of this plan;
  that's more for formatting user messages,
  which I'm going to not prioritize further testing of for now (we have a few rudimentary golden tests).

* Modify `exec.cpp` to take an array of logged array elements (var name + idxs).
  Upon each Fence/Arrive/Await,
  log the before and after synchronization environment state and
  (if applicable) arrive count of each logged array element.
  Maybe JoinThreads as well; up to you.
  Decide the best behavior for not-allocated/out-of-range logged array elements.
  Note: don't worry about the alpha-rename-per-loop-iteration stuff in amspork3.
  I do that to avoid having to define a "clear environment" rule which would mess up the monotone proof.
  For the testing, just log for any live allocation of that name.

* I would prefer a text format stored to disk similar to how excut works
  (excut is still useful for Sm80/Sm90a "execute cuda tests", just not camspork)
  where the reference and actual traces are inspectable after the test is run.
  BONUS: include Stmt IDs in the trace and translate to Exo proc file/line with `get_stmt_srcinfo`.

* The `syncv_table` is templatized on whether SyncvLogRequest is given.
  Test the NO LOG path; this is the binary that is actually being run.
  NOTE: don't test specifically with the "single sync var" mode
  (or at least not solely with this)
  even if the trace is only for one array element as bugs in hash tables
  and copy-on-write are part of the attack surface to test.
  i.e. keep the new logging entirely based on query-syncv code in the exec.cpp layer.
  
* Test tricky cases: Write procs, execute them in camspork with some logged elements,
  and use the Python references to generate expected reference traces.
  Compare that the traces match.
  I don't care about the exact ordering of visibility records or de-duplication
  (i.e. compare things as sets).
  Look at `test_coll_units.py` for various complicated patterns,
  test non-contiguous (strided) thread collectives,
  test weird `CudaWarps` including the tricky name restriction + non-trivial numeric range restriction case,
  test a bunch of different async instructions,
  test atomics,
  test the AssignmentRecord copy-on-write functionality in multiple ways,
  e.g. by accessing a tensor via a big window followed by a bunch of small windows (test de-convergence),
  test the opposite (test not incorrectly merging state),
  test access by partially overlapping windows.
