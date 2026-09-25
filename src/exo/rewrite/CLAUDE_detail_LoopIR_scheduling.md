* `DoUnsafeRemoveIf(stmt_c, recursive=True)`: fixed 2026-09-25 in `exo`.
  Before, each child was recursed on with a cursor into the *original* root, so only the last child of each body kept its edits (earlier sibling `if`s survived silently), and the forwarding `_compose` sat outside the loop.
  Now child cursors are forwarded through prior siblings' edits.
  `with` statements (smuggled as `LoopIR.If`) are kept: recursive mode descends into them, and a non-recursive cursor to one raises.
  Tests: `tests/test_schedules.py::test_unsafe_remove_if_*`.
  No generated-code change for the current Sm90 gemm; the wgmma-zero prototypes in [plan_wgmma_zero.md](../../../plan_wgmma_zero.md) relied on the old bug to keep `if iter_k == 0` and would need explicit guard removal.
