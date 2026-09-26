# Plan: Zero-Init wgmma Instrs (remove `wgmma_zero`)

Status: implemented 2026-09-25 (`exo` `aad79e3a`; see "Implementation notes" at the end).

Depends on: nothing.
Enables: [plan_static_bool_folding.md](plan_static_bool_folding.md) (the deferred "non-constant scale-d" warning).
Related: [plan_seq_unroll_head.md](plan_seq_unroll_head.md) (peeling iteration 0, which ptxas requires for good code),
[plan_remove_vis_flags.md](plan_remove_vis_flags.md) and [plan_proxy_fence_rules.md](plan_proxy_fence_rules.md) (both simplify once `wgmma_zero_qual` is gone).
Research record and rejected alternatives (incl. ThunderKittens-style `mm`/`mma`): [plan_wgmma_zero_alternatives.md](plan_wgmma_zero_alternatives.md).

## Goal

"The first wgmma of an accumulation uses `scale-d = 0`" is currently modeled by a separate instr, `Sm90_tk_zero_scale_d`.
The instr sets a flag in the tile struct, and sync-check models it with a fake write-only mutate (`wgmma_zero_qual`).
That fake write hides a prior CUDA access to the D registers, so this passes sync-check without the required `wgmma.fence`:

    CUDA write of D -> Fence(cuda_in_order, cuda_in_order) -> Sm90_tk_zero_scale_d(D) -> wgmma(D)

(Reproduced 2026-09-25; see the table in the research record.)

Replace this with wgmma instrs whose `behavior` states the zero-init directly, so it goes through normal instr unification and sync-check.
Then delete the zero instr, the QualTL, and the tile-struct flag.

## Design

### Instr families (per A/B mode: `row_col`, `rmem_col`, `col_col`, `row_row`, `rmem_row`, `col_row`)

1. **Zero-init form** (runtime `zero_init: bool`; `scale-d = !zero_init` on native k-step 0, `1` afterwards):

        def behavior(N: size, K: size, D: [R][4, 16, N], A: ..., B: ..., zero_init: bool):
            for m_warp in seq(0, 4):
                for mi in seq(0, 16):
                    for n in seq(0, N):
                        if zero_init:
                            D[m_warp, mi, n] = 0.0
                        for k in seq(0, K):
                            D[m_warp, mi, n] += A[...] * B[...]

2. **Accumulate-only form** (hardwired `scale-d = 1`; no `if` in `behavior`), so code that never zero-inits can `replace` without a dummy `if False:` in the pre-substitution code.
   Its `behavior` is identical to today's `Sm90_tk_mma_*`.

Naming: the accumulate-only form keeps today's `Sm90_tk_mma_{a}_{b}` names.
Its behavior doesn't change, so existing callers that never used the zero instr are unaffected.
The zero-init form gets a suffix, e.g. `Sm90_tk_mma_{a}_{b}_zi`.

David Zhao Akeley: `_zi` naming is approved.

Keep **one top-level statement** in every `behavior` (the zero stays nested inside the `n` loop).
`DoReplace` swaps `len(behavior.body)` statements for one call, so a two-statement behavior would need 2-statement block cursors.
Cursors into the second statement would also stop forwarding (`_forward_replace`, `core/internal_cursors.py`, the 2025-10-24 comment).

Put `zero_init` last among the runtime args (open), so handwritten calls read `Sm90_tk_mma_row_col_zi(D[...], A[...], B[...], iter_k == 0, D=f32, ...)`.
It is not listed in `instance()`, so it is a runtime arg, not a template parameter.
Unification already supports bool "holes" (`unify_bool_hole`, `rewrite/LoopIR_unification.py`).
The matched condition may only use variables free in the replaced block, e.g. `iter_k == 0` bound outside the wgmma loop nest.

### Codegen (`platforms/Sm90/Sm90_tk_mma_impl.py`)

`make_basic_mma(a_mode, b_mode, zero_init_form: bool)`:

* accumulate-only: `ptx.add_arg(1, constraint="n", log_as=None, N=1)` (an int is pasted into the PTX as a true constant).
* zero-init: native k-step 0 passes `int(!(<zero_init C expr>))` with constraint `"r"`; the `int(...)` is required because a C++ `bool` fails `"r"`. Later k-steps pass the constant `1`.
* No `D.index(get_scale_d=True)` and no `scale_d = 1;` reset.
* Once [plan_static_bool_folding.md](plan_static_bool_folding.md) lands, a statically known `zero_init` becomes a literal, and a non-constant one warns (deferred).

### Deletions

* `gen_Sm90_tk_mma.py` / `Sm90_tk_mma.py`: delete `Sm90_tk_zero_scale_d`; stamp out both families (12 instrs).
* `kittens_impl/tk_types.py`: delete `exo_CudaTkScaleD`, its `MemGlobalC`, `get_scale_d`, and `.tile`.
  Tiles allocate as plain `::kittens::rt_*<...>`, which changes every `CudaTkWarpTile` allocation (goldens, see Tests).
* `spork/timelines.py`: delete `wgmma_zero_instr`, `wgmma_zero_qual`, the `cuda_rmem_qual_tl_dict` entry, the `cuda_basic_instr_tl` entry, the `_cuda_temporal_quals` entry, and the `generate_latex_table` row.
  Regenerate `spork_b/gSyncTL.tex`. The qual-bit count drops by one.
* `Sm90_fwd.py`: drop the two re-exports.
* `Sm90_TkRmemTileD.qual_tl_dict`: `{wgmma_async_instr: wgmma_async_rmem_d_qual}`.

The mm/mma prototype patch already contains most of these deletions and can be mined.

### `schedule_gemm` (`platforms/Sm90/tk_gemm_util.py`)

Keep the source spec (`D_rmem = 0; for k: D_rmem += ...`, with `assert K_cluster > 0`).
Right after `divide_loop(k -> iter_k, sub_iter_k)`, sink the zero into the `iter_k` loop:

    gemm = add_loop(gemm, D_zero, "iter_k", f"(K_cluster + {smem_K - 1}) / {smem_K}", guard=True)
    gemm = fuse(gemm, iter_k_loop.prev(), iter_k_loop)   # fuse deletes the 2nd loop; forward the 1st

`add_loop(guard=True)` proves the trip count positive from `assert K_cluster > 0`.
Then:
* drop the prologue `fission(gap_before_main)` and the "Finalize zero prologue" block;
* re-anchor `gap_after_main` on the fused loop;
* replace the `mw` loop with the zero-init form.

This yields `zero_init = (iter_k == 0)`.

**`unsafe_remove_if` workaround (required):**
`unsafe_remove_if(gemm, wgmma_ms_cursor, True)` now really removes every nested `if` (bug fixed in `exo` `77808786`), including `if iter_k == 0`.
Replace that call with a local helper in `tk_gemm_util.py` that removes only the M/N/K tail guards:
walk the `ms` subtree and call non-recursive `unsafe_remove_if` on each `If` except the zero guard, identified structurally (the `If` whose body is the `D_rmem = 0` assignment).
No `exo` API change needed.
The other four `unsafe_remove_if` call sites only touch tail guards (log of all 5 calls, 2026-09-25) and are unaffected.

### Handwritten `tk_gemm_util` paths and other callers

* `handwrite_row_{col,row}_gemm`: delete the `Sm90_tk_zero_scale_d` blocks.
* `main_loop`s (row_col coop, row_row coop, row_row A-in-rmem, ping-pong): wgmma calls become the zero-init form with `iter_k == 0`.
  `sched_cut_sync_iter_k` still peels iteration 0, so nvcc folds the flag.
* `tests/cuda/test_misc_cuda_err.py::mkproc_packed_dims_point_expr_err` used the zero instr only as a distributed-dims instr; substitute another one.
* `sporkbench/examples/Sm90a_tk_fa` (**needs human buy-in**; read-only repo): 1 zero + 1 mma → the zero-init form with `hdim64 == 0`.
  Its `hdim64` loop is fully unrolled (`pragma_unroll=0` emits `#pragma unroll`), so the flag should fold.
* The Sm90a gemm sporkbench examples need no edits.

## Tests / validation

* Sync-error regression for the hole (table from the research record):
  * `wgmma.fence` → pass;
  * `Fence(cuda_in_order, cuda_in_order)` → WAW;
  * no fence → WAW.
* Handwritten style: a zero-init call with an `hdim64 == 0` argument passes sync-check.
* Goldens: `tests/golden/cuda/test_Sm90a_gemm/*` and any other `CudaTkWarpTile` allocations; regenerate and review.
* `exocc` for all `Sm90a_*_schedule_row_col_gemm` and handwritten sporkbench examples, with separate output dirs per example (camspork JIT dir race).
* `nvcc -arch=compute_90a -code=sm_90a ... -Xptxas -v`: no C7515 (C75xx diagnostics only print with `-v`).
  In SASS, each accumulation's first HGMMA is `RZ, !UPT` (checked 2026-09-25 for the prototypes).
  The A-in-rmem kernel's C7513 predates this work.
* `gen_Sm90_tk_mma.py` must reproduce `Sm90_tk_mma.py` verbatim (it was stale until `exo` `b22b1a53`).

## Performance note

ptxas does **not** peel iteration 0 on its own.
A runtime `zero_init` (or `if k == 0: ...`) in an unpeeled loop gets C7515 "wgmma serialized" (toy experiments in the research record).
Today `cut_loop` provides the peel; long term, [plan_seq_unroll_head.md](plan_seq_unroll_head.md) does.
The deferred warning in [plan_static_bool_folding.md](plan_static_bool_folding.md) exists to catch forgetting it.

## Implementation notes (2026-09-25)

Landed as designed. Deviations and details:

* `gen_Sm90_tk_mma.py` stamps out the accumulate-only family (variants 1-6, unchanged names) then the `_zi` family (7-12); `make_basic_mma(a_mode, b_mode, zero_init_form)`.
* `schedule_gemm`: the selective guard removal is `remove_tail_guards_keep_zero(p, cursor)` in `tk_gemm_util.py`.
  `gap_before_main` is gone along with the prologue fission.
* `test_misc_cuda_err.py::mkproc_packed_dims_point_expr_err` now uses `cuda_tk_tile_zero` on the `mi` loop (warp instr; needs a `simplify` after `replace` to drop `mw + 0`).
* New `tests/cuda/test_Sm90a_wgmma_accum.py` (after landing): accumulate-only wgmma on D preloaded from GMEM, for all 6 A/B modes (8 dtype variants), golden (+ nvcc `sm_90a`) and Sm90a runtime vs. numpy.
  Loads, stores, and the wgmma are `replace`d from plain loop nests.
  H100 (human, 2026-09-25): all 8 runtime variants pass (`exo` `99b0ebda`).
* New `tests/cuda/test_wgmma_zero_init_sync.py`: the hole table (wgmma fence pass; `cuda_in_order` fence and no fence both WAW) and the handwritten `hdim64 == 0` loop (sync-check + compile).
* `timelines.generate_latex_table` had a hardcoded `wg0` column header; fixed.
  Its output is split by hand: the key lines go to `spork_b/QualTL.tex`, the tabular to `spork_b/gSyncTL.tex` (both regenerated).
* Goldens regenerated: `test_Sm90a_gemm/*` (tile type, no zero prologue, runtime scale-d only on k-step 0) and `test_3cycle_mbarrier/*` (only `wgmma_zero_qual` lines removed from sync-error dumps).
* `sporkbench/examples/Sm90a_tk_fa` edited (human approved): zero + mma → `Sm90_tk_mma_row_col_zi(..., hdim64 == 0, ...)`.

Validation on this (sm_80) machine:

* `exocc` succeeds (including built-in `sync_check`) for all 36 `Sm90a_*` + `unflash_attn` sporkbench examples.
* nvcc `sm_90a` (sporkbench's flags, `-Xptxas -v`): all 36 compile.
  The only C75xx is the pre-existing C7513 on `..._m1n2_m256n192_coop_splitK_Armem` (A-in-rmem).
* `Sm90a_tk_fa` SASS: the QK^T accumulation starts with one `RZ, !UPT` HGMMA; the other HGMMAs accumulate (nvcc folded `hdim64 == 0` after unrolling).
* SASS spot check (`m1n2_coop_os_Sffff`, `m1n2_coop_os_Hbbff`): each accumulation's first HGMMA is `RZ, !UPT`, all later ones accumulate; 168 registers, no spills.

H100 (human, 2026-09-25): `Sm90a_tk_fa`, `Sm90a_row_major_gemm`, and the Sm90a runtime tests pass with the new code.
