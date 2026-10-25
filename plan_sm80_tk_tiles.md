# Plan: Sm80 MMA on ThunderKittens Tiles

Status: research done 2026-10-23, nothing implemented.

Depends on: nothing.
Research record: [plan_sm80_tk_tiles_ldsm_exp.cu](plan_sm80_tk_tiles_ldsm_exp.cu) (TK ldmatrix/mma SASS experiment, 6 variants)
and [plan_sm80_tk_tiles_shuffle_check.py](plan_sm80_tk_tiles_shuffle_check.py) (SASS checker, proposed as the regression test).

## Goal

`exo.platforms.Sm80` MMA currently uses three unrelated register representations:

* A/B: `CudaRmemPacked32` arrays with hand-chosen dims (`[8, 4, 2, 2, K_pack]` for A, `[8, 4, 2, K_pack]` for B).
* C/D: the opaque `Sm80_RmemMatrixD(16, 8)`.
* Its epilogue instrs are all legacy `Sm80_mma_*_d_*_tf32` instrs.

(Note: the D operand is *not* currently ThunderKittens; only `Sm90` uses TK D tiles.)

Replace all of it with `CudaTkWarpTile` (`kittens::rt`) tiles, so Sm80 code shares the tile ops, vec ops, and
GMEM/SMEM copies already wrapped in `exo.platforms.cuda_tk`. That makes `CudaRmemPacked32` a non-MMA type only.

Hard requirement: no silent degradation to `ldmatrix` output that ptxas must shuffle before `mma`.

## Findings

### TK register layout in MMA coordinates

For a 16×16 `rt_base`, in *MMA* coordinates (`mn` = M for A / N for B, `k` = reduction dim),
TK's `data[q]` holds the 8×8 block with `mn_hi = q & 1`, `k_hi = q >> 1`. This holds for **both** TK layouts:

| tile | TK dims | `data[1]` | `data[2]` |
|---|---|---|---|
| row layout, A (`m × k`) or B for `mma_ABt` (`n × k`) | rows = mn | rows + 8 | cols + 8 |
| col layout, A for `mma_AtB` (`k × m`) or B for `mma_AB` (`k × n`) | rows = k | cols + 8 | rows + 8 |

(Read off TK's non-Hopper `store` paths in `ops/group/memory/tile/shared_to_register.cuh`.)

`mma.m16n8k16` consumes these register vectors (TK `ops/group/mma/warp.cuh` agrees):

* A: `{data[0], data[1], data[2], data[3]}`: a 4-register aligned quad in SASS.
* B: `{data[0], data[2]}` for n 0–7 and `{data[1], data[3]}` for n 8–15: two aligned register pairs.
* D (f32): `{data[0].x, data[0].y, data[1].x, data[1].y}` and `{data[2].*, data[3].*}`. These are read-modify-write by the mma itself, so loads never conflict with them.

So **the A role wants the PTX `ldmatrix` vector in order `{d0,d1,d2,d3}`, and the B role wants `{d0,d2,d1,d3}`**, whatever the tile layout.
`ldmatrix` writes a contiguous SASS quad (`LDSM.16.M88.4 R44`), so the order is fixed at the PTX vector.
Any mismatch costs moves.
This is the r1/r2 transposition failure.

### Stock TK gets two of the four cases wrong

TK `warp::load(rt, st)` (16-bit) always emits `ldsm4(d0,d1,d2,d3)` for row layout and `ldsm4t(d0,d2,d1,d3)` for col layout.
That is right for (A, row) and (B, col), and wrong for (B, row) and (A, col).
The wrong (B, row) case is K-major B, i.e. `mma_ABt`: exactly the layout of the sporkbench GEMM (`B: f16[L, N, K]`).

Measured on sm_80 (nvcc 12.3, `-O3`; one warp, 16×64 · 64×64 per iteration; all variants numerically exact vs a CPU reference):

| variant | load | mma | moves reading an LDSM result |
|---|---|---|---|
| V1 | TK row A, TK row B | `mma_ABt` | **49** |
| V2 | TK row A, TK col B (`ldsm4t`) | `mma_AB` | 0 |
| V3 | TK row A, row B via **permuted-address** `ldsm4` | `mma_ABt` | 0 |
| V4 | = V1 with hand-written A load (sanity) | `mma_ABt` | **49** |
| V5 | TK col A (`ldsm4t`), TK col B | `mma_AtB` | **60** |
| V6 | col A via **permuted-address** `ldsm4t`, TK col B | `mma_AtB` | 0 |
| — | current Exo `starter_ring_smem_gemm` (sporkbench, 250 regs) | | 0 |

V1 SASS shape: `LDSM.16.M88.4 R44, [...]`, then `IMAD.MOV.U32 R40, RZ, RZ, R44; MOV R41, R46; IMAD.MOV.U32 R46, RZ, RZ, R45`, then `HMMA ... R40 ...`, for every B `ldmatrix`.

### The fix is purely in the address mapping

`ldmatrix` lets lane group `s` (lanes `8s..8s+7`) point at any 8×8 block, and block `s` lands in PTX output slot `s`.
So bind the output slots in *consumer* order and pick addresses to match:

* slot `s` holds `data[q(s)]` with `q = s` for the A role and `q = [0, 2, 1, 3][s]` for the B role;
* lane `l` (`s = l / 8`, `r = l % 8`) addresses block `q(s)`, which is at
  (tile row, tile col) = `(8·mn_hi + r, 8·k_hi)` for row layout (`ldmatrix`), or `(8·k_hi + r, 8·mn_hi)` for col layout (`ldmatrix.trans`).

That's V3/V6. The current Exo `Sm80_ldmatrix_base` `operand="A"/"B"` path already does this for `CudaRmemPacked32`.
The new code must keep doing it, with the tile type deciding instead of a free-form argument.

Correctness never depends on the permutation, only performance.
That is exactly why it fails silently.
Golden tests (PTX text) can't catch it either.
Hence the SASS test below.

### tf32 is out (confirmed, plus a second reason)

* TK itself: `rt_base` for sm_80 only allows `bf16_2`, `half_2`, `float2`, and TK has no tf32 `mma`.
* Fundamental: an `rt_fl` thread holds K columns `{2t, 2t+1}` (one `float2`), but the `m16n8k8.tf32` A/B fragment needs columns `{t, t+4}`.
  Those are *different elements per lane*, so a true-semantics tf32 path on TK tiles needs cross-lane shuffles, not just moves.
* The escape hatch is a consistent K permutation of both A and B (valid for a dot product).
  But then the tile's contents aren't the matrix in TK's coordinates: TK stores, transposes, and column reductions would silently be wrong.
  So that would be a separate opaque, non-TK memory type, i.e. what we're deleting.

If tf32 is ever wanted again, revive it as its own opaque type outside this standardization. Not planned.

### Other things found

* `Sm80_mma_m16n8k8_s32_s32` emits `mma...m16n8k8.row.col.s32.s32.s32.s32`, which ptxas rejects ("Unexpected instruction types specified for 'mma'").
  It has never been usable; delete it. (Integer mma is s8/u8/s4/b1.)
* `Sm80_cp_async_2d` declared `src` as `CudaSmemAtomicity16B` instead of `CudaGmemAtomicity16B`.
  Fixed and tested independently of this plan (`exo` `0d70b28a`, `test_cp_async_2d`).
* `cuda_tk_load_rs` (wraps TK `warp::load` from swizzled SMEM) is fine for Sm90 `rmem` A, which is the (A, row) case.
  Feeding a Sm80 B operand with it would hit the V1 failure, so it must not be the Sm80 operand-load path.

## Design

### Memory types (Sm80 module)

Role-tagged subclasses of `CudaTkWarpTile`, mirroring `Sm90_TkRmemTileA/D`:

* `Sm80_TkRmemTileA(M, K, layout)`: `layout="row"` gives `rt<M, K, row>` (from K-major SMEM); `"col"` gives `rt<K, M, col>` (from M-major SMEM).
* `Sm80_TkRmemTileB(N, K, layout)`: `"row"` gives `rt<N, K, row>` (K-major, `mma_ABt`); `"col"` gives `rt<K, N, col>` (`mma_AB`).
* D: plain `CudaTkWarpTile(M, N, "row")` (f32 or f16); no tag needed, since nothing order-sensitive writes it except mma.

The role tag is the static guard against silent shuffles: the mma instrs set `access_info[...].mem` to the tagged type, so passing an A-loaded tile as B is a type error.
Storage is identical to the untagged tile, so TK tile ops still apply.
To verify: whether Exo's memory checks accept the subclass wherever an instr asks for the base `CudaTkWarpTile(r, c, layout)` (Sm90 D tiles presumably already rely on this).

Retagging (e.g. flash-attention `P = convert(S)` from D to A) goes through a TK copy/convert instr.
D→A is order-compatible (D's `data[q]` is also `m_hi + 2·n_hi`), so f32→f16 conversion is in-place per `data[q]`.
Verify this in SASS once an attention-style test exists.

### Instrs

1. **`Sm80_tk_ldmatrix`**: warp copies a whole tile SMEM → `Sm80_TkRmemTile{A,B}`.
   `behavior`: `dst[i, j] = src[i, j]` in the tile's own coordinates with `stride(src, 1) == 1`.
   Row vs col TK layout then simply means plain vs `.trans`.
   Codegen: one `ldmatrix.x4` per 16×16 base tile, with addresses from the formula above.
   Use `InlinePtxGen` with explicit `{tile.data[q(0)], ..., tile.data[q(3)]}` (`*reinterpret_cast<uint32_t*>(&...)`, as Sm90 does), so the vector order is visible in goldens.
   SMEM: `CudaSmemLinear` and `Sm90_SmemSwizzled(32|64|128)` via per-lane `index_ptr`, as the current `Sm80_ldmatrix` does.
   Addressing: compute the swizzled per-lane base once per call, then use `[reg + imm]` for period-multiple sub-tile offsets and per-k-chunk XOR variants.
   Don't recompute `S(base + offset)` per `ldmatrix.x4`.
   Keep the lane-dependent part separable so Category 1 can hoist it.
   See [plan_sm80_pointer_research.md](plan_sm80_pointer_research.md) for how CUTLASS does this.
   Types: f16, bf16. (8-bit types need `rt` packing we don't have on sm_80; skip.)
2. **`Sm80_tk_mma_{AB, ABt, AtB, AtBt}`** (or one instr family parameterized by A/B layout, generated like `gen_Sm90_tk_mma.py`):
   `D[M, N] += A × B` on whole tiles, `M, N, K` multiples of 16.
   Codegen loops over base tiles, emitting two `mma.sync.aligned.m16n8k16` per (m, n, k) base triple with the vectors listed above.
   Use `InlinePtxGen` with `volatile=False` (as the current `Sm80_mma_m16n8` does).
   This is correct because the asm is a pure register op with honest in/out constraints.
   Reordering happens mostly in ptxas regardless, so this is cheap perf insurance, not a measured win.
   TK's `hmma16816` is `asm volatile`.
   Types: (f32; f16, f16), (f32; bf16, bf16), (f16; f16, f16).

### Decision: generate ldmatrix and mma PTX ourselves, not via TK (agreed 2026-10-23)

TK provides only register *storage* (`rt` types) plus the existing elementwise/reduction tile ops.
`ldmatrix` and `mma.sync` PTX are generated in Exo, as Sm90 already does for TMA and wgmma. Reasons:

* The ldmatrix lane-address permutation is only correct relative to the mma's consumed register order.
  Both must live in one Exo file (ideally one shared table), not split across Exo and a TK header.
  If TK changed its order, we'd get exactly the silent failure: right numbers, extra moves.
* TK's `warp::load` is wrong for 2 of the 4 operand cases anyway, so the load side must be ours regardless.
* The operand vectors are visible in generated `.cu` and goldens, so an r1/r2 swap shows up in review.
* HazyResearch no longer maintains sm_80 TK paths; don't build on them beyond the `rt` storage layout.
  That layout is shared with the Hopper and Blackwell paths, so it's safe.
* SASS cost is identical either way (V3 used TK's mma with zero shuffles), so this is about maintainability and safety, not speed.
3. Zero / epilogue: reuse `cuda_tk_tile_zero`, `cuda_tk_store_rg` (GMEM), `cuda_tk_store_rs` (SMEM).
   No new Sm80-specific epilogue instrs.

TK tiles can't be windowed below the whole tile (`packed_tensor_shape = (r, c)`).
So mma/ldmatrix granularity is "whole tile", and users allocate arrays of tiles at the granularity they want to schedule.
N = 8 tiles are gone (TK minimum is 16).
The sporkbench GEMM uses 64×64 warp tiles, so that's no loss.

### Deletions

From `Sm80.py`: `Sm80_ldmatrix_{base,16b,32b,f32,f16,bf16,i32}`, `Sm80_mma_m16n8*` (all 5 partials), `Sm80_mma_m16n8_zero`, `Sm80_mma_zero_d_tf32`,
`Sm80_BasicRmemMatrix`, `Sm80_RmemMatrixD*`, `AB_ptx_names`/`CD_ptx_names`, and all six `Sm80_mma_*_d_*_tf32` epilogue instrs plus `_codegen_Sm80_d_tf32`.
Keep `cp.async`, `Sm80_CommitGroup`, timelines.
`CudaRmemPacked32` + `cuda_packed32_*` stay (used by non-MMA tests in `test_cu_f16.py`); only its "abandoned" comment changes.

## Steps

0. Baseline: run sporkbench `pldi_Sm80_gemm` and record TFLOPS, registers (250 now), and the shuffle check (0 now).
1. Add `plan_sm80_tk_tiles_shuffle_check.py` to `exo/tests` as a helper.
   Add a test that compiles a CUDA test kernel to an sm_80 cubin (`$EXO_NVCC -cubin`, no GPU needed), runs `cuobjdump -sass`, and asserts zero shuffles.
   Seed it with the V1/V3 kernels to prove it fails and passes as intended.
2. Add the tagged memory types + `Sm80_tk_ldmatrix` + `Sm80_tk_mma_*`.
   Tests: numeric tests for all 4 (A layout × B layout) combos × 3 type triples × {`CudaSmemLinear`, swizzle 128};
   the SASS shuffle test on each combo; and a negative test that A-tagged tiles are rejected as B.
3. Port the sporkbench `pldi_Sm80_gemm` `starter_ring_smem_gemm` (needs your buy-in: sporkbench edit).
   Roughly `D: f32[2, 2, 64, 64] @ CudaTkWarpTile(64, 64)`, `A_rmem: f16[2, 2, 2, 64, 16] @ Sm80_TkRmemTileA(64, 16, "row")`, same for B, one mma per k16 step, `cuda_tk_store_rg` epilogue.
   Accept: exact results, ≤ 250 regs and no spills, 0 shuffles, TFLOPS ≥ baseline.
   (The commented-out `starter_*` variants in that file can be ported or dropped.)
4. Port tests: `test_coll_units.py` (uses `Sm80_RmemMatrixD(16, 8)` only as a generic warp tile, so use `CudaTkWarpTile(16, 16)` + `cuda_tk_tile_zero`/`cuda_tk_store_rg`).
   Rewrite `test_simple_cuda.py::xgemm_Sm80_fence` (the only tf32 user) as an f16 GEMM.
   Regenerate and review goldens.
5. Delete the legacy code listed above. Update docs outside `spork_b` that show the old API (`spork/docs/nexo*`, `nyc25*`, `f16_gemm_*`, `Sm80_tmp.py`, `su_samples*`, `tmem_todo.tex`); `spork_b` has no Sm80-MMA material.

## Open questions

* Role tags vs. an `operand=` argument: tags are safer but cost a retag instr in attention-style code. I recommend tags.
* We depend on TK's `rt_base` `data[q]` layout, but that layout is effectively ABI-stable.
  It's the same storage the Sm90 wgmma D/A wrappers and TK's Blackwell RMEM/TMEM ld/st rely on, so no pinning is needed for this plan.
  Only TK's sm_80-specific instr wrappers (`warp::load` ldmatrix paths, `hmma16816`) are at risk, and this plan doesn't use them.
  The step-2 numeric tests (Exo ldmatrix → mma → TK store vs reference) cover the layout as a side effect.
* Separate tech debt (not this plan): the local TK (`$EXO_KITTENS`) has David's ad-hoc modifications, so there's no public commit to cite.
  An older public commit probably suffices; identify one eventually so the build is reproducible.
* Should `Sm80_tk_ldmatrix` also accept Sm90 targets? On sm_90 the same instrs are valid, and TK's Hopper path is identical for loads.
