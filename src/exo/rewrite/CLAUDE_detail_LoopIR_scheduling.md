* `DoUnsafeRemoveIf(stmt_c, recursive=True)` bug: the child loop reassigns `ir, fwd_child` from each child's cursor on the *original* root, so only the last child of each body keeps its edits.
  Earlier sibling `if`s survive silently (repro: `for i: (if a: ...); (if b: ...)` removes only `if b`).
  Also, `fwd = _compose(fwd_child, fwd)` sits outside the loop.
  Note: recursive removal would also strip `with CudaWarps` contexts (they are `LoopIR.If`), if it worked.
  `schedule_gemm` relies on this in at least the wgmma-zero prototype ([plan_wgmma_zero.md](../../../plan_wgmma_zero.md)).
