# Research record: `wgmma_zero` alternatives

Status: research record (2026-09-25), kept for re-evaluation. **Not the plan.**
The decided plan is [plan_wgmma_zero.md](plan_wgmma_zero.md) (zero-init + accumulate-only wgmma instrs).
This file keeps the original problem statement, the human's first idea (`if k == 0` in the source spec), the unevaluated alternatives,
the `_zi` prototype, the ThunderKittens-style `mm`/`mma` prototype, the sync-hole reproduction, and the ptxas peeling experiments.
Prototype diffs: [plan_wgmma_zero_prototype.patch](plan_wgmma_zero_prototype.patch) (`_zi`, against `exo` `84ef21b4`)
and [plan_wgmma_zero_mm_prototype.patch](plan_wgmma_zero_mm_prototype.patch) (`mm`/`mma`, against `exo` `b22b1a53`).
Both prototypes relied on the since-fixed `unsafe_remove_if` bug (`exo` `77808786`).

Original title follows.

# Plan: `wgmma_zero` Hole

Status: researched 2026-09-25 (see "Research findings" at the end); recommended fix prototyped, not landed. Found 2026-09-25.
Intended to be delegated to a separate agent researching fixes in the scheduling/rewrite language (not the sync system).

## The real problem

`wgmma_zero` has nothing to do with synchronization.
It exists because "the first wgmma of an accumulation uses `scale-d = 0`" was piggybacked onto the sync-check system (via `wgmma_zero_qual`)
instead of being expressed through the Exo rewrite system, where it properly belongs
(e.g. an instr/rewrite relating `D = 0; D += A @ B` to a single `scale-d = 0` wgmma, checked by the normal equivalence machinery).
The sync hole below is a symptom. A fix in the rewrite language should let `wgmma_zero_qual` and its special cases in the tile `qual_tl_dict`s be deleted.
Starting points: `rewrite/LoopIR_scheduling.py`, `rewrite/LoopIR_unification.py`, `platforms/Sm90/Sm90_tk_mma_impl.py` (`Sm90_tk_zero_scale_d`), `tk_gemm_util.py`.

Related: [plan_remove_vis_flags.md](plan_remove_vis_flags.md) (wgmma register QualTLs).

## Background

`wgmma_zero_instr` (e.g. `Sm90_tk_zero_scale_d`) does not touch registers.
It sets an "IOU" that makes the *next* `wgmma.mma_async` use `scale-d = 0` instead of `scale-d = 1`.
Sync-check models this with `wgmma_zero_qual`:

* `cuda_rmem_qual_tl_dict[wgmma_zero_instr] = [wgmma_zero_qual, wgmma_async_rmem_d_qual]`
* `Sm90_TkRmemTileD` maps `wgmma_async_instr` to `[wgmma_async_rmem_d_qual, wgmma_zero_qual]`, so a real wgmma accepts a D tile whose last "write" was the zero.
* Today `wgmma_zero_qual` is a temporal-only member of `cuda_in_order` etc. (to go away per [plan_proxy_fence_rules.md](plan_proxy_fence_rules.md);
  replacement: put `cuda_in_order_rmem_qual` in `wgmma_zero`'s precondition set).

## The hole (exists today, and is not closed by the planned changes)

    CUDA access to D registers        (record: cuda_in_order_rmem_qual)
    wgmma_zero on D                   (write-only mutate: REPLACES the mutate record with (g, wgmma_zero_qual))
    real wgmma on D                   (precondition includes wgmma_zero_qual → passes)

No `wgmma.fence` between the CUDA access and the real wgmma is required by sync-check, because the zero's (fake) write hides the CUDA access.
PTX requires `wgmma.fence` between a register access by the warpgroup and a subsequent `wgmma.mma_async` accessing the same registers
(except accumulator accesses across wgmma of the same shape).

In practice no example relies on the hole: every known example has a `Fence(wgmma_fence_1, wgmma_fence_2)` between any CUDA access and the real wgmma.
(Correction 2026-09-25: only `Sm90a_tk_fa` issues it *before* the zero; `tk_gemm_util.py`, both handwritten and scheduled, issues the zero first and the wgmma fence after it, inside the main loop.)

## Human-generated Idea (unevaluated)

The primary purpose of Exo-GPU is to talk about the synchronization
work, so I'm OK with a cheesy solution for fixing the scheduling.
Currently, only `tk_gemm_util.py` actually uses the Exo scheduling
language to substitute in wgmma instructions (specifically
`schedule_gemm`) and all other code just "handwrites" the wgmma code,
so it won't be affected by any scheduling or wgmma instr changes (fact
check this).

Either by hand-written modification or scheduling (bonus points),
modify the starting `gemm` proc in `schedule_gemm` to have

    D_rmem: f32
    # D_rmem = 0  (removed)
    for k in seq(0, K_cluster):
        if k == 0:  # Added
            D_rmem = 0
        D_rmem += A_tensorMap[batch, m, task_k, k] * B_tensorMap[batch, n, task_k, k]

NB the `K_cluster > 0` is critical for this to work
(0 init is skipped if `K_cluster == 0`, therefore garbage is written).

Modify all wgmma instrs to have `behavior` pseudocode

    for m:
        for n:
            if zero_init:  # !scale-D, but Exo doesn't parse `not`...
                D[m, n] = 0
            for k:
                D[m, n] += A[m, k] * B[n, k]

Eliminate the `bool scale_d` (or whatever it's called) in the
thunderkittens tile struct I hacked together.

Check if all the rewrites preserve the structure in a form where the specified new wgmma instr can be substituted in.

BTW if this is actually implemented (I am not confident at all this is
a good idea), also introduce alternative wgmma Exo instrs that hard
wire scale-D=1 and dispense with the `if zero_init:`.
Since the libs are autogenerated with the `gen_` scripts,
stamping out 2x the instrs should be easy.


## Alternative Ideas (unevaluated)

* Model `wgmma_zero` as not a mutate at all: a no-op on the sync env, with the `scale-d = 0` effect attached to the next real wgmma
  (requires sync-check to know the zero and the wgmma are paired; they are separate instr calls in LoopIR).
* Make the zero a mutate that *keeps* prior mutate records (like atomics do: `r* ∪ ρ(x).m`), so the next real wgmma still has to satisfy the CUDA access.
* Remove `wgmma_zero` as a user-visible instr; generate `scale-d = 0` from a flag/argument on the real wgmma instr.
  The human "knew wgmma_zero was a bad idea".


## Research findings (2026-09-25)

Prototype code lived on a throwaway `exo` branch, which was deleted afterwards; the diff is saved as [plan_wgmma_zero_prototype.patch](plan_wgmma_zero_prototype.patch) (applies to `exo` `84ef21b4`).

### Recommendation

Adopt a variant of the human idea: give the wgmma instr a runtime `zero_init: bool` argument and delete `Sm90_tk_zero_scale_d`.
Do **not** move the zero into the `k` loop of the source spec; do the rewrite at `iter_k` granularity with existing primitives (details below).
Still stamp out the hardwired-`scale-d = 1` variants (the prototype already generates them as `_acc`).
Per the human, they are for schedules where D already holds real data and there is no zero-init to match, not for efficiency.
(Efficiency alone wouldn't justify them: nvcc folds the flag; see "Codegen".)

Behavior of the prototyped `Sm90_tk_mma_{a}_{b}_zi` (generated by `gen_Sm90_tk_mma.py` with `scale_d_mode="arg"`):

    def behavior(N: size, K: size, zero_init: bool, D: ..., A: ..., B: ...):
        for m_warp in seq(0, 4):
            for mi in seq(0, 16):
                for n in seq(0, N):
                    if zero_init:
                        D[m_warp, mi, n] = 0.0
                    for k in seq(0, K):
                        D[m_warp, mi, n] += A[...] * B[...]

* `zero_init` is not listed in `instance()`, so it is a runtime arg, not a template parameter (`tparams_from_signature`, `core/instr_class.py`).
* Unification: `LoopIR_unification.py` already supports `bool` "holes": `if zero_init:` unifies with any `if <cond>:` whose `cond` only uses variables free in the replaced block (`unify_bool_hole`).
  `replace` is sound by construction (structural match + solved args), so there is no extra equivalence check to satisfy.
* Codegen: native k-step 0 passes `int(!(<zero_init C expr>))` as the `scale-d` operand; later k-steps pass `1`.
  (The `int(...)` cast is required: a C++ `bool` fails the `"r"` asm constraint.)
  No `exo_CudaTkScaleD::scale_d` access, no `scale_d = 1` reset.
* Sync-check: `D` is read (`+=`) so it is a normal mutate with precondition `wgmma_async_rmem_d_qual`; there is no fake write-only record.

### Scheduling (`schedule_gemm`)

The `k == 0` form in the source has the wrong granularity.
After `divide_loop(k → iter_k, sub_iter_k)` the condition becomes `32 * iter_k + sub_iter_k == 0` *inside* the `sub_iter_k` loop, and this is the loop that the instr's `for k` consumes.
The condition depends on `sub_iter_k`, which is bound inside the replaced block, so `unify_bool_hole` refuses it.
Going back to the right shape would need peeling plus "un-peeling", and there's no primitive for the second step.
(This is by reasoning; I did not run the source-level variant.)

What works is to keep the source spec unchanged (`D_rmem = 0; for k: D_rmem += ...`) and, right after `divide_loop`, sink the zero into the `iter_k` loop:

    gemm = add_loop(gemm, D_zero, "iter_k", f"(K_cluster + {smem_K - 1}) / {smem_K}", guard=True)
    #   -> for iter_k in seq(0, ...): if iter_k == 0: D_rmem = 0
    gemm = fuse(gemm, <that loop>, iter_k_loop)
    #   -> for iter_k: (if iter_k == 0: D_rmem = 0); for sub_iter_k: ...

* `add_loop(guard=True)` runs `Check_IsPositiveExpr` on the trip count. This is where the source's `assert K_cluster > 0` is load-bearing, confirming the human's NB.
* The prologue fission (`fission(gap_before_main)`) and the whole "Finalize zero prologue" block go away; the zero rides along through `lift_alloc` / `lift_scope` / `expand_dim` / `stage_mem`.
* Resulting pre-replace shape under the `ms` loop is `for mw: for mi: for sub_cta_n: (if iter_k == 0: D = 0); for sub_iter_k: D += ...`, which unifies with `_zi` and yields `zero_init = (iter_k == 0)`.
* `sched_cut_sync_iter_k` still cuts the loop *after* `replace`, so both halves carry the arg `iter_k == 0`. That's harmless (see Codegen).
* **Caveat (bug since fixed in `exo`, 2026-09-25; the prototype now needs explicit guard removal):** the prototype survives `unsafe_remove_if(gemm, wgmma_ms_cursor, True)` only because of a bug.
  Recursive `DoUnsafeRemoveIf` only keeps edits from the last child of each body (`rewrite/LoopIR_scheduling.py:2386`), so it strips the K tail guard (the last child) but misses `if iter_k == 0` (the first child).
  A proper implementation should remove the M/N/K guards explicitly, or fix the bug and make removal selective.
  If the bug is fixed naively, `replace` fails loudly (an `Assign` doesn't unify with an `If`), so nothing goes silently wrong.

Results with `WGMMA_ZERO_RESEARCH=1` (f32, m1n2, r4, both `m128n256` and `m256n192`, both `os` and `splitK`):

* `schedule_gemm` completes, including its built-in `sync_check`, and `exocc` succeeds.
* The generated `.cuh` differs from baseline only in: the zero-prologue block is gone (along with two now-empty no-op CTA loops), `D_rmem[ms].scale_d` becomes `int(!((iter_k == 0)))` or `1`, and the `scale_d = 1;` resets are gone.
* nvcc `sm_90a` compiles with no ptxas "wgmma serialized" diagnostics (baseline has none either).
  Rechecked with `-Xptxas -v`: C7513/C7515 are `ptxas info` messages and only print with `-v`.
* SASS: every first HGMMA of an accumulation is `..., RZ, !UPT` (scale-d = 0), and all others accumulate, exactly like baseline.
  nvcc folds `iter_k == 0` completely in both halves of the cut loop.
  Total SASS count is 3640 vs 3704 (baseline); HGMMA count is 24 vs 28 (a different unroll in the m128n256 kernel, not investigated); registers 168 both.
  Can't run on this machine (sm_80 only).

### Sync-check hole: reproduced and closed

Test: one warpgroup does a `cuda_tk_tile_zero` of D, then an optional fence, then either (old) `Sm90_tk_zero_scale_d` + `Sm90_tk_mma_row_col`, or (new) `Sm90_tk_mma_row_col_zi(True, ...)`, then commit/wait.

| fence between CUDA write and wgmma | old (zero instr) | new (`_zi`) |
|---|---|---|
| `Fence(wgmma_fence_1, wgmma_fence_2)` | pass | pass |
| `Fence(cuda_in_order, cuda_in_order)` | **pass (the hole)** | WAW (correct) |
| none | WAW | WAW |

So the hole needs a non-wgmma fence (e.g. `__syncthreads`-like) before the zero.
With no fence it is caught, because the zero's own precondition `{wgmma_zero_qual, wgmma_async_rmem_d_qual}` rejects the CUDA record.
I also checked the neighboring "the zero is a semantic lie" cases; today's rules catch all of them:

* zero → (any fence) → CUDA **read** of D: RAW, because `wgmma_zero_qual` is temporal-only and reads need full visibility.
* zero → (none / in-order / wgmma fence) → CUDA **write** of D → `wgmma.fence` → wgmma: WAW.
  This matters because Exo semantics would say `D = written + AB` while hardware computes `AB` (the flag is still 0).

Handwritten style also works: `Sm90_tk_mma_row_col_zi(hdim64 == 0, D[...], ...)` inside a loop with CUDA writes of D between batches passes sync-check, and nvcc accepts the codegen.

### What a full implementation would touch

* `gen_Sm90_tk_mma.py`: emit only the `_zi` flavor, or rename it to the existing names; delete `Sm90_tk_zero_scale_d`.
  **Pre-existing bug:** the generator is stale relative to its output.
  Commit `e6743eff` (2026-03-25) hand-edited `Sm90_tk_mma.py` (zero codegen `…index(get_scale_d=True)`), but the generator still emits `{args.D.index()}.scale_d = 0;`, which is now broken since `index()` returns `.tile`.
  Rerunning the generator used to regress this. Fixed in `exo` `b22b1a53`: the generator now reproduces its output verbatim.
* `Sm90_tk_mma_impl.py`: drop the `scale_d_mode` scaffolding; D tile `qual_tl_dict` → `{wgmma_async_instr: wgmma_async_rmem_d_qual}`.
* `kittens_impl/tk_types.py`: delete `exo_CudaTkScaleD`, `get_scale_d`, and the `MemGlobalC`.
  This changes every kittens tile's C++ type, so it touches `exo/tests/golden/cuda/test_Sm90a_gemm/*` (6 files) and anything else that allocates `CudaTkWarpTile`.
  Aside: the struct both inherits from `Tile` *and* has a `Tile tile` member, which looks unintended (the inherited base is unused).
* `spork/timelines.py`: delete `wgmma_zero_instr`, `wgmma_zero_qual`, its `cuda_rmem_qual_tl_dict` entry, the `cuda_basic_instr_tl` entry, and its line in `generate_latex_table` (so regenerate `gSyncTL.tex`).
  The camspork qual-bit count drops by one.
  This also simplifies [plan_proxy_fence_rules.md](plan_proxy_fence_rules.md) / [plan_remove_vis_flags.md](plan_remove_vis_flags.md), since there's no longer a temporal-only special case to replace.
* `tk_gemm_util.py`: scheduled path as prototyped; handwritten paths (`handwrite_*_gemm` zero blocks + `main_loop` wgmma calls) pass `iter_k == 0`, as the zero moves into `main_loop` (not done in the prototype).
* `tests/cuda/test_misc_cuda_err.py::mkproc_packed_dims_point_expr_err` uses `Sm90_tk_zero_scale_d` only as a convenient distributed-dims instr; it needs a substitute.
  Add the hole table above as a sync-error regression test.
* `sporkbench/examples/Sm90a_tk_fa/exocc_Sm90a_tk_attn_fwd.py`: **needs human buy-in** (read-only repo). 1 zero + 1 mma call become `Sm90_tk_mma_row_col_zi(hdim64 == 0, ...)`.
  Its `hdim64` loop is `pragma_unroll=0`; folding the flag there depends on nvcc unrolling it anyway, which is no worse than today's runtime `scale_d` member.
  The Sm90a gemm sporkbench examples need no edits; they only call `schedule_gemm` / `handwrite_gemm`.

Fact check of the human claim "only `tk_gemm_util.py` uses scheduling to substitute wgmma": **true.**
`replace(..., Sm90_tk_*)` appears only in `schedule_gemm` (and the test above).
Handwritten wgmma calls: `tk_gemm_util.py` handwritten paths, `Sm90a_tk_fa`.
There are no other wgmma instr families (`wgmma_async_instr` is only used by `Sm90_tk_mma*`).

### Alternatives (brief evaluation)

* *No-op zero, pair it with the next wgmma in sync-check:* the `_zi` arg is exactly this pairing, done statically at the instr level, so sync-check needs no new concept.
* *Zero keeps prior mutate records (atomic-style union):* closes the sync hole, but keeps an instr whose `behavior` (`D = 0`) doesn't match what it does.
  Today's temporal-only trick is the only thing stopping CUDA reads/writes after it from being wrong (see above), and that trick is scheduled for deletion.
* *Flag/arg on the real wgmma:* this is the recommendation.

### Open question: `_zi` arg vs. ThunderKittens-style `mm` / `mma` split

Human preference (2026-09-25): hardwired variants, `mm` (scale-d = 0, behavior `D = 0; D += AB` with no `if`) and `mma` (scale-d = 1), match the SASS reality better (two distinct HGMMA forms, `RZ, !UPT` vs. accumulate) than a "runtime" flag that must fold to a constant anyway.
Scheduling implication (untested): `replace` would have to happen after the iter_k-0 peel, not before as today.
Plan: cut `iter_k` at 1 first. In the `seq(0, 1)` half, `if iter_k == 0` must fold to unconditional; in the `seq(1, ...)` half, it must be eliminated (`eliminate_dead_code` should manage this with range info).
Only then substitute `mm` / `mma`.
Tested on a toy proc (`cut_loop(iter_k, 1)` over `for n: (if iter_k == 0: D[n] = 0); D[n] += ...`):
`simplify` does **not** fold either `if` (and per the human, making it do so would break unseen, fragile schedules).
An explicit `eliminate_dead_code(p, <the if>)` **does** fold both: in the `seq(0, 1)` half it becomes an unconditional `D[n] = 0`, and in the `seq(1, K)` half the `if` is deleted.
So mm/mma needs one opt-in `eliminate_dead_code` per half before `replace`, with no change to `simplify`.
Not yet tried inside the real `schedule_gemm` nest, where the `if` sits under `cuda_threads` loops and the zero stays inside a 1-trip `seq(0, 1)` loop.
A `_zi` flavor could still be kept for handwritten code that doesn't peel.


## mm/mma investigation (2026-09-25, later)

Prototype: [plan_wgmma_zero_mm_prototype.patch](plan_wgmma_zero_mm_prototype.patch) (applies to `exo` `b22b1a53`; throwaway branch deleted).
It removes `scale_d` completely and converts all users in `exo`, except `tests/cuda/test_misc_cuda_err.py` (not updated) and `Sm90a_tk_fa` (read-only).

### Library side: straightforward

* `gen_Sm90_tk_mma.py` stamps out `Sm90_tk_mm_{a}_{b}` (behavior `D = 0; for k: D += AB`) and `Sm90_tk_mma_{a}_{b}` (`D += AB`) for all 6 A/B modes, via `make_basic_mma(a_mode, b_mode, zero_init)`.
  `Sm90_tk_zero_scale_d` is gone.
* Codegen: scale-d is a compile-time constant passed with `ptx.add_arg(int, constraint="n", N=1)`.
  Ints are pasted in as "true constants", so the PTX reads `setp.ne.b32 p, 0, 0;`.
  It is 0 only for native k-step 0 of `mm`; the PTX format and excut logging shape are unchanged.
* `tk_types.py`: `exo_CudaTkScaleD`, `get_scale_d`, and the `MemGlobalC` are deleted; tiles allocate as plain `::kittens::rt_*<...> name[...]` and `index()` returns the tile itself.
* `timelines.py` / `Sm90_fwd.py`: `wgmma_zero_instr`, `wgmma_zero_qual` and all their entries deleted.
  The D tile `qual_tl_dict` becomes `{wgmma_async_instr: wgmma_async_rmem_d_qual}`.
  Nothing else in `src/` referenced them, and camspork has no hardcoded reference.

### `schedule_gemm`: works, replace moves after the cut

Same zero sinking as before (`add_loop(guard=True)` + `fuse`), but no `replace` before the cut.
After `sched_cut_sync_iter_k`, `eliminate_dead_code` runs on every `if iter_k == 0` (new helper `sched_fold_iter_k_0`).
Then each `for mw` loop is replaced with `mm` if its enclosing `iter_k` loop has `lo == 0`, else `mma`.
The fence/arrive around the `ms` loop is inserted before the cut on the not-yet-substituted nest, which works fine.
The same `unsafe_remove_if` bug caveat as before applies.

f32 m1n2 os + splitK: `sync_check` and `exocc` pass, and nvcc compiles cleanly (no C75xx with `-v`).
SASS: 3 `RZ, !UPT` HGMMAs, 168 regs, 3624/3608 total instrs (baseline 3704/3696, `_zi` 3640/3632).

### Handwritten `tk_gemm_util`: works with an explicit if/else

Each handwritten wgmma call (4 sites: row_col coop, row_row coop, row_row A-in-rmem, ping-pong) becomes

    if iter_k == 0:
        Sm90_tk_mm_X(...)
    else:
        Sm90_tk_mma_X(...)

The separate zero blocks in `handwrite_row_{col,row}_gemm` are deleted.
The cost is duplicated call args in the handwritten text; there is no macro mechanism in `@proc`.
The optional "sneaky scheduling" is just `sched_fold_iter_k_0` at the end of `sched_cut_sync_iter_k`.

bf16 row_col coop m1n2, ping-pong m2n1 (the human says the ping-pong code is bogus anyway; results reported but not relied on), row_row coop m1n2 splitK, row_row A-in-rmem:
all pass sync-check and compile.
**With or without the fold, SASS is identical per kernel**: once LoopIR has the cut, nvcc resolves the if/else by itself (1 `RZ` HGMMA each, 2 ARRIVEs per 8 HGMMA).
The A-in-rmem kernel gets ptxas C7513 ("non wgmma instructions defining *input* registers"), but that is **pre-existing** on unmodified `akeley98/endgame` (the `cuda_tk_load_rs` into A registers).

### Relying on downstream compilers without any peel: does NOT work (ptxas)

Toy (sync-checked): one warpgroup, runtime `K`, loop `for k: Fence(wgmma); if k == 0: mm else: mma; Arrive; Await(cg, N)`, then a `cuda_tk_store_rg` of D.

| variant | ptxas `-v` | SASS HGMMA / WARPGROUP.ARRIVE |
|---|---|---|
| no peel, Await 0 | **C7515 serialized** | 22 / 22 |
| no peel, Await 1 | **C7515 serialized** | 22 / 22 |
| no peel, Await 1, `#pragma unroll 1` | **C7515 serialized** | 9 / 9 |
| no peel, Await 1, `#pragma unroll 4` | **C7515 serialized** | 38 / 38 |
| manually peeled (mm outside, `seq(1, K)` mma loop), Await 0 / 1 | clean | normal |

C7515 is "non wgmma instructions defining accumulator registers": ptxas does not peel iteration 0 on its own.
This matches `spork_b/wgmma_dirty_laundry.tex` ("Special Casing Iteration 0").
A *runtime* scale-d flag in an unpeeled loop (today's `scale_d` member, or `_zi`) would be no better; it's the same pattern.
So, whatever replaces `cut_loop`, **something upstream of ptxas must peel iteration 0 on the consumer path**: LoopIR scheduling, or a codegen-level "peel first iteration" seq option.
An improved `LoopIR_compiler` range analysis could only fold the `if` once the peel exists; it can't create the peel.

### Without `cut_loop`: `specialize` + `eliminate_dead_code` + `lift_scope`

In case the LoopIR-level cut goes away (e.g. replaced by a consumer-only codegen peel), the scheduled path can still reach `mm`/`mma` without cutting.
Tested on a toy `for iter_k: for n: (if iter_k == 0: D[n] = 0); for sub_k: D[n] += ...`:

1. `specialize(p, n_loop.body(), "iter_k == 0")` produces `if iter_k == 0: (body) else: (body)`.
2. `eliminate_dead_code` on the inner `if iter_k == 0` in each branch folds it: it **uses the enclosing path condition**, so the zero becomes unconditional in the then-branch and is deleted in the else-branch.
3. `lift_scope` on the outer `if/else` hoists it above the `n` loop: `for iter_k: if iter_k == 0: (for n: D = 0; for sub_k ...) else: (for n: for sub_k ...)`.
   In the real nest that means lifting through `sub_cta_n`, `mi`, `mw` (3 lifts).

Each branch then matches `mm` / `mma` directly.
Codegen then relies on the (hypothetical) consumer peel to make the `if` static for ptxas; per the table above it must not be left as a runtime branch.
Not tried inside the real `schedule_gemm` nest.

### Remaining work for a real implementation

* `tests/cuda/test_misc_cuda_err.py::mkproc_packed_dims_point_expr_err` needs a replacement distributed-dims instr.
* Goldens: every `CudaTkWarpTile` allocation changes type (`test_Sm90a_gemm` goldens at least); not run.
* `Sm90a_tk_fa` (needs buy-in): `if hdim64 == 0: mm else: mma`.
  Its loop is `pragma_unroll=0`, which is `#pragma unroll` (full unroll) with the `Hdim/64` trip count, so the `if` should fold after unrolling (not tested).
