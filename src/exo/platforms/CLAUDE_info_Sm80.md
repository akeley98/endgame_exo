# `exo/src/exo/platforms/Sm80.py`

sm_80 (Ampere) memory, instrs, and timelines; every exported name contains `Sm80_`.

* `cp.async`: `Sm80_cp_async_{1d,2d,f32}`, `Sm80_CommitGroup`, instr-tl `Sm80_cp_async_instr`.
* `ldmatrix`: `Sm80_ldmatrix_{16b,32b}` into `CudaRmemPacked32`. The `(nmat0, nmat1) = (2, 2)` path takes `operand="A"|"B"`,
  which permutes lane addresses + output registers so the PTX vector matches what `mma` consumes (no SASS register shuffles).
* `mma.sync.m16n8k{8,16}`: `Sm80_mma_m16n8`, with A/B in `CudaRmemPacked32` and C/D in opaque `Sm80_RmemMatrixD(16, 8)` (not ThunderKittens).
* Legacy `Sm80_mma_*_d_{row,col}_major_tf32` epilogue instrs (despite the name, used for f16 GEMM epilogues too).

Known defects:

* `Sm80_mma_m16n8k8_s32_s32` emits PTX that ptxas rejects (no s32 A/B mma exists).
* (Fixed in `0d70b28a`) `Sm80_cp_async_2d` used to declare `src` as `CudaSmemAtomicity16B`; it's now GMEM and covered by `test_cp_async_2d`.

Plan [plan_sm80_tk_tiles.md](../../../../plan_sm80_tk_tiles.md): replace the ldmatrix/mma/RmemMatrixD/epilogue parts with role-tagged
ThunderKittens tiles (`Sm80_TkRmemTile{A,B}`), `Sm80_tk_ldmatrix`, `Sm80_tk_mma_*`; drop tf32 and s32.
