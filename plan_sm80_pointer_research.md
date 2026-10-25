# Research: How CUTLASS Sm80 GEMM Generates ldmatrix / cp.async Addresses (vs Exo)

Status: research record, 2026-10-24. No code changes.
Feeds: Category 1 "Pointer Arithmetic" in [CLAUDE.md](CLAUDE.md), and the `Sm80_tk_ldmatrix` codegen in [plan_sm80_tk_tiles.md](plan_sm80_tk_tiles.md).

## Kernels compared

Same problem shape: f16 × f16 → f32, A row-major and B column-major (both K-major), 128×128×32 CTA tile, 64×64 warp tiles, `mma.m16n8k16`, 4 stages.

* CUTLASS: sporkbench `runner/sporkbench_cutlass_Sm80_f32_f16.cu` (CUTLASS 2.x `device::Gemm`, `MmaMultistage`).
  SASS from the existing sporkbench build (`examples/pldi_Sm80_gemm/bin/.../sporkbench_cutlass_Sm80_f32_f16.sm_80.cubin`), CUTLASS checkout `endgame_exo/cutlass` @ 3f5bafb3.
* Exo: sporkbench `pldi_Sm80_gemm/exocc_Sm80_starter.py` `starter_ring_smem_gemm`, compiled with `exocc` + nvcc 12.3 `-O3 -arch=sm_80`.

Instantiated CUTLASS types (demangled):
gmem→smem `PredicatedTileAccessIterator` + `RegularTileAccessIterator<..., RowMajorTensorOpMultiplicandCrosswise<16, 32>>`
(B: `ColumnMajor...Crosswise<16, 32>`), thread map `PitchLinearWarpRakedThreadMap<32×128, 128 thr, <4,8>, 8>`,
and warp-level `MmaTensorOpMultiplicandTileIterator<..., TensorOpMultiplicandCrosswise<16, 32>, ...>` (`gemm/warp/mma_tensor_op_tile_iterator.h:2149`).

## Main loop, one 128×128×32 K-tile per iteration

| | CUTLASS | Exo |
|---|---|---|
| registers | 230 | 250 |
| instrs in main loop | 162 | 259 |
| useful (64 HMMA + 16 LDSM + 8 LDGSTS) | 88 (54%) | 88 (34%) |
| vector int ALU (IADD3/IMAD/LOP3/LEA/SHF) | 39 | 162 |
| uniform-datapath ops (UIADD3/UMOV/UISETP) | 18 | 0 |
| distinct LDSM address registers (whole kernel) | 4 | 17 |
| LDSM / LDGSTS using `[reg + imm]` (whole kernel) | 21/24, 28/32 | 0/24, 0/32 |

About 20 of CUTLASS's overhead instrs are bounds-predicate logic (ISETP/LOP3/SEL on residue masks) that Exo doesn't need, which makes the gap in address math wider than the totals suggest.
These are static counts over the loop body, not a dynamic profile. The 15–30% loss in `spork/docs/nexo.tex` is consistent with 1.6× the issue slots at equal HMMA count.

CUTLASS main-loop addressing (excerpt):

    LDSM.16.M88.4 R164, [R2+UR4] ;            // A, k-group 1: per-lane R2 + uniform stage offset
    LDSM.16.M88.4 R184, [R2+UR4+0x1000] ;     //   next 16 rows: immediate
    UIADD3 UR8, UR4, 0x80, URZ ;              // next stage = +128 B (uniform)
    @!UP0 UIADD3 UR8, UR4, -0x180, URZ ;      // ring wrap 3 -> 0 (uniform; counter UR18 vs 4)
    LEA R214, R198, UR8, 0x4 ;                // A, k-group 0: per-lane R198 (vector index) * 16 + stage
    LDSM.16.M88.4 R132, [R214+0x1000] ;
    @P0 LDGSTS... [R217], [R200.64] ;         // R217 = R196 + UR12 (per-thread smem base + uniform write stage)
    @P6 LDGSTS... [R219+0x8000], [R202.64] ;  // B buffer: immediate
    IADD3 R224, P6, R200, c[0x0][0x188], RZ ; // gmem pointer += precomputed per-K-tile byte increment

Exo, per LDSM (all of it is recomputed for every ldmatrix):

    LEA R174, P2, R169, R210, 0x1 ; LEA.HI.X R175, R169, R211, ... ;  // 64-bit generic address = base + offset
    SHF.R.U64 R172, R174, 0x3, R175 ;                                   // swizzle(addr): addr ^ ((addr >> 3) & 0x70)
    LOP3.LUT R172, R174, 0x70, R172, 0x78, !PT ;
    IADD3 R172, R172, -c[0x0][0x18], RZ ;                               // generic -> shared
    LDSM.16.M88.4 R172, [R172] ;

## Checking David's analysis

1. **"CUTLASS uses a position-independent swizzle `base + swizzle(offset)`; `Sm90_SmemSwizzled` is `swizzle(base + offset)`. Not a big deal."**
   Confirmed: `TensorOpMultiplicand::operator()` (`layout/tensor_op_multiplicand_sm75.h:151`) maps a logical coord to a swizzled element offset, added to `ref.data()`.
   `Sm90_smem.py` says outright that it's a position-*dependent* swizzle of the raw address (`swizzle_pointer`).
   Agreed it's not a big deal. Correction (2026-10-25, brute-forced for x ∈ {32, 64, 128}, S(a) = a ^ ((a >> 3) & mask)):
   * **Hoisting needs no base alignment in either scheme.** PDSL: S(a + kP) = S(a) + kP for *every* a, so `S(base + lane_off)` can be hoisted and stepped by kP for any base.
     PISL: the same identity on the offset; base never enters S.
     P is also the *smallest* step that commutes for all a. (An earlier version of this note wrongly said hoisting needs `base mod P`.)
   * **Bucket = x bytes.** S only rewrites bits below log2(x), so S(a) stays in the same x-aligned group as a, and S permutes each group.
     For **PDSL** this is a bounds-safety requirement: an allocation stays inside itself under S for all sizes iff `base % x == 0` (and its size is a multiple of x).
     That's why `Sm90_SmemSwizzled.smem_config` forces alignment = `swizzle`, and it suffices.
     **PISL** needs no base alignment for bounds (offsets in [0, n·x) stay in [0, n·x)), only the 16 B alignment the accesses need anyway.
   * **PISL ≡ PDSL** (`base + S(o) == S(base + o)`) iff `base % P == 0`. x-alignment is not enough.
     This matters only when mixing the two on one buffer, e.g. a PISL software layout read by hardware that swizzles absolute addresses (TMA/wgmma).
     It doesn't matter for hoisting within Exo's PDSL.

2. **"swizzle(ptr + i) != swizzle(ptr) + i in general."** Confirmed. Nothing below the period commutes additively in general.

3. **"Each swizzle has a period P; if P divides the byte offset, it commutes."** Confirmed. This is the `spork_b` glossary entry gSwizzlePeriod.
   Exo's code has `swizzle_period = 8 * swizzle`: for `Sm90_SmemSwizzled(x)`, P = 8·x bytes (1024/512/256), matching David's math.
   Masks: `addr ^ ((addr >> 3) & {0x70, 0x30, 0x10})`, i.e. bits [4..6] / [4..5] / [4] XOR bits [7..9] / [7..8] / [7].

4. **"CUTLASS precomputes a pre-swizzled per-lane pointer and strides by multiples of the period."** Confirmed, with three refinements:

   * **XOR steps within a period.** A second identity applies besides the period one.
     If an offset `c` sets only *destination* bits of the XOR (the 16 B-chunk index, bits [4..6] for 128 B), then `S(y ^ c) = S(y) ^ c`.
     CUTLASS uses it for K steps inside a swizzle row: `add_tile_offset` does `byte_offset_ ^= k_groups_delta * 32` (bytes; `mma_tensor_op_tile_iterator.h`, Crosswise specialization).
     XOR can't go in an addressing-mode immediate.
     So in SASS this becomes one precomputed per-lane register per k-group: `R2`/`R198` for A and `R0`/`R197` for B.
     That's cheap because there are only `kGroupsPerTile` (here 2) of them.
   * **Uniform registers for the warp-uniform part.** The stage offset lives in `UR4`/`UR5` (write side `UR12`/`UR14`).
     It advances by `+0x80` with a conditional `-0x180` wrap, never `(k % RING) * stride`.
     Only the lane-dependent part sits in vector registers, computed once before the loop.
     This ties into the `exo_warpid` item: the uniform datapath is what makes this nearly free.
   * **Immediates for the rest.** Sub-tile offsets that are period multiples (16 rows = 0x1000 B here) are folded into `[reg + imm]`.
     CUTLASS also folds the A/B buffer split (0x8000).

   One layout detail differs from Exo, though the conclusion doesn't depend on it.
   This CUTLASS layout (`kFactor = 2`: two 64 B K-rows per 128 B line, stages placed side by side along K, 512 B per row-pair) works out to element (row, 16 B-chunk c, stage s) at
   `512·(row/2) + 128·s + 64·(row%2) + 16·(c ^ ((row/2) % 4))`,
   i.e. XOR address bits [4,5] with bits [9,10] (CuTe `Swizzle<2,4,5>`, not a TMA mode).
   The +128 B stage step touches bits [7,8], which are neither XOR source nor destination, so it commutes even though it's smaller than the 2 KiB period.
   Exo doesn't need this. With stages in an outer dimension (as `starter_ring_smem_gemm` already does: 8 KiB per stage), the stage step is a multiple of P = 1024.

5. **"We can't do this if the Exo ldmatrix applies its own per-thread offset at the last moment, deep in `InstrInfo.codegen()`."**
   Mostly confirmed. The real constraint is which parts codegen can *see*:
   * *Within one instr call*, codegen can do CUTLASS's trick by itself: one swizzled per-lane base, then immediates for period-multiple sub-tiles and XOR variants for in-row chunks.
     The planned whole-tile `Sm80_tk_ldmatrix` (4 × `ldmatrix.x4` per 64×16 tile) should do this. It needs only the window's strides and P, which `WindowFeatures` already has.
   * *Across calls and loop iterations* (ring stage, K-tile, warp offsets), codegen sees one call at a time and emits `S(base + everything)`.
     nvcc/ptxas can't factor that, because they don't know the XOR/shift is periodic.
     So this needs the window itself to carry `(pre-swizzled per-lane 32-bit pointer, additive period-multiple offset)`, hoisted out of the loop.
     That's the "`Sm90_SmemSwizzle` window encoder with swizzled and unswizzled 32-bit pointer" item in the Category 1 notes.
     Moving ldmatrix's per-lane offset out of `codegen` (the `inline_codegen` → LoopIR-nest idea) is what lets the per-lane part be hoisted like any other loop-invariant value.

## Additional Exo costs found (not in the original analysis)

* **64-bit generic pointers.** Exo swizzles the 64-bit generic address (`SHF.R.U64`, `LEA.HI.X`) and converts generic→shared on every access (`IADD3 -c[0x0][0x18]`).
  CUTLASS works in 32-bit shared addresses throughout. Swizzling a shared-window address is only equivalent if the generic shared window base is P-aligned (it is in practice), and it costs 64-bit ops either way.
* **GMEM side of cp.async.** CUTLASS keeps one 64-bit global pointer per access, bumped by a precomputed per-K-tile byte increment (kernel param `c[0x0][0x188]`).
  Exo recomputes `base + (ks·32 + ...)·2` each time. This is ordinary (unswizzled) pointer-increment strength reduction, the simplest Category 1 case.

## Implications

* `Sm80_tk_ldmatrix` (plan_sm80_tk_tiles.md): emit `S(base + lane_off)` once per call, then `+imm` for period-multiple row offsets, and per-k-chunk XOR variants as separate registers.
  Keep the lane-dependent part a pure function of `(lane, swizzle, strides)` so it can later be hoisted.
* Category 1 metric: report main-loop instr count / useful fraction / `[reg+imm]` ratio from the SASS (the tables above) as the before/after measure. Target ≈ CUTLASS's 162 per 128×128×32 K-tile.
* No new alignment requirement for hoisting: `S(base + lane_off) + k·P` is exact for any base under Exo's PDSL.
  The existing x-byte alignment of `Sm90_SmemSwizzled` (`Sm90_smem.py`, "Force 32, 64, or 128 byte alignment") is what keeps PDSL accesses in bounds, and it's already enforced.
