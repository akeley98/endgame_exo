# Fix Missing Atomics in QtlMask

Need to fix the complaint mentioned in [plan_camspork_test.md](plan_camspork_test.md).
Update the linked complaint when this task is done.

* `exo/src/exo/core/memory.py` has a function `MemWin.make_qual_tl_mask` that creates
  a bitfield representation of what `spork/docs/amspork3` denotes as `\sporkQtlMask`.
  This is supposed to give all qualitative timelines applicable to the memory type,
  but the automatic deduction misses atomic qualitative timelines.

* Update the `Qual_tl` type to have a `QualMaskClass` enum.
  Four choices: `data_ram` (CPU, random addressable types: GMEM/SMEM), `cuda_rmem`, `mbarrier`, `tcgen05`
  (even though tcgen05 turned out to be vaporware for this project).
  The `if True:` code that creates the `Qual_tl` constants needs to have 2 lines per defined constant now to not be too wide, but don't re-enable the black formatting.

* Assert each `Qual_tl` has mask class `data_ram` iff its proxy class is not `neither`.
  Background: this is the reason the mask exists.
  For analyzing the trace graph of data accesses, we need to be able to cleanly separate observers
  as generic or async, so we can identify the vertices that must correspond to proxy fences.
  This is undermined if the graph can form edges that are neither "generic" nor "async".

* `MemWin` has to have a new member func that gives the `QualMaskClass` (default: not implemented error).
  Find all the memory base classes that opt into sync checking by having `sync_exempt() -> False`.
  Update these classes to give a real `QualMaskClass`.
  Then update `make_qual_tl_mask` to give a mask of all qtl that have said mask class.

* This will change behavior for wgmma timelines (will appear in all register memory masks). This is completely fine.

* Remove the `atomic_qual_mask` exemption in camspork's `syncv_table.cpp`:
  `CAMSPORK_REQUIRE_CMP((access.qual_tl_mask & access.precondition_qual_bits & ~atomic_qual_mask),`
