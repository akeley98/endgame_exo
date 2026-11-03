# Plan: Derived Windows as Codegen-Only Aliases (`derived` attribute)

Status: research prototype 2026-11-02, committed as exo `2439b40c` on branch `akeley98/derived_window` (not merged)
(snapshot: [plan_derived_window_prototype.patch](plan_derived_window_prototype.patch)).
Verdict so far: feasible; generated code unchanged; see Open Decisions before implementing for real.

Depends on: nothing.
Related: [CLAUDE.md](CLAUDE.md) "Pointer Arithmetic" (WindowStmt is the only IR-level way to compute an offset once);
[plan_seq_unroll_head.md](plan_seq_unroll_head.md) (alpha-renames window statements in copies; `Alpha_Rename` must rename `derived`);
`spork/docs/SpecialWindow.tex` (legacy doc; its Proposal 2 is the discarded approach revived here);
`exo/tests/test_BUGGY_WindowStmt.py` (known WindowStmt bugs).

## Proposal (David Zhao Akeley)

* `WindowStmt` stays. A new rewrite inserts a `WindowStmt` with no other changes.
* Nothing in LoopIR indexes the lhs of a `WindowStmt`.
  Accesses index the underlying ("root") tensor, and may name in `derived` the `WindowStmt` they should be expressed through.
* Just before codegen, each access with `derived` is unified against its `WindowStmt` and becomes what it would be today.
* Rewrites pass `derived` through un-inspected, except as required by the Sym rule below.
* Stale `derived` is always diagnosed, but late (at unify) is OK.

## Rules (agreed 2026-11-02)

* **Sym and `derived` are one unit.**
  A rewrite that changes the Sym of an access is responsible for setting `derived` to a correct value:
  `None`, or a renamed window (e.g. `Alpha_Rename`, `inline` rename both the WindowStmt and `derived` consistently).
* **Stale `derived` is an error, never silently dropped** (`DerivedWindowError` from unify).
  Silently dropping would hide exactly the pointer-arithmetic regressions we care about.
* **No containment proof** for interval coordinates: TMA tail tiles routinely extend past the `Sm90_tensorMap` window
  (relying on the unsound `AccessInfo.allow_out_of_bounds` hack, now commented as such in the code).
  For plain windows, an access outside the window but inside the tensor is the same address anyway.
* **Point coordinates** of the WindowStmt must equal the access's: affine normalization (prototype), SMT with context possible later.
* **Config reads are banned in window statement indices** (typecheck error; also in `insert_window_stmt`).
  Desugaring moves index computation from the WindowStmt to each use, which is wrong if the config changes in between.
  Codegen never supported config reads there anyway (`AssertionError('bad case!')` today).

## Deviation From the Original Sketch

`derived` must also be on `Read`, `Assign`, `Reduce`, not only `WindowExpr`.
Otherwise element accesses through a window desugar to base-tensor indexing and codegen recomputes the window offset on every access
(observed: `d_A_tile.data[...]` became `d_A[(... + 128 * m_cta) * K + ...]` in `test_cu_f16`),
and window structs captured by device functions disappear (defeats `test_grid_constants_windows`).

LoopIR grammar (prototype):

    Assign( sym name, type type, expr* idx, expr rhs, sym? derived )
    Reduce( sym name, type type, expr* idx, expr rhs, sym? derived )
    Read( sym name, expr* idx, sym? derived )
    WindowExpr( sym name, w_access* idx, sym? derived )

Constructor churn was mechanical (~50 positional sites got `None`; `.update` copies `derived` for free).
Failure mode for a rewrite that rebuilds a node and forgets `derived`: plain window → silently slower code (not caught by unify);
special window → `mem_analysis` error (wrong memory for the instr argument).

## Pipeline (prototype, `core/derived_window.py`)

* `desugar_derived_windows`: right after typecheck (`API.Procedure.__init__`), and on the body spliced in by `DoInline`
  (window arguments become WindowStmts there).
  `w[i]` → `x[chained] @ derived(w)`; bare `w` call arg → WindowExpr of the rhs; `stride(w, d)` → `stride(x, d')`.
* All analyses (`CheckBounds`, new_eff, `mem_analysis`, `CollAnalysis`, `sync_check`) see the desugared form.
* `unify_derived_windows`: in `run_compile`, after `run_backend_checks`, before `Compiler` (hence before `loopir_lower_cuda`).
  Lexically scoped lookup (so `cut_loop`-style duplication with shared Syms works).
* Printing: `x[1 + 2, 4:12] @ derived(w)`, reads parenthesized. Print-only; the parser does not accept it.
  Object code still writes `w[i]` (frontend sugar).

## Code That Must Inspect `derived` (prototype list)

* `Alpha_Rename` (Read/WindowExpr/Assign/Reduce `derived`).
* `DoInline`: desugar the alpha-renamed body (window args → WindowStmt).
* `DoInlineWindow`: now just deletes the WindowStmt and redirects `derived=w` to `w`'s own rhs `derived`.
* `DoReplace`: copy `derived` from the matched accesses to the new WindowExpr args (Sym unchanged);
  else search in-scope WindowStmts for a unique `SpecialWindow` matching the instr param memory (see Open Decisions).
* Sym-changing rewrites clear `derived`: `stage_mem`, `unroll_buffer`. `reuse_buffer` keeps it (renames WindowStmt rhs too).
* Node rebuild sites that must copy `derived`: `expand_dim` (Read/WindowExpr), `split_write`, `fold_into_reduce`, `prec_analysis`, `cuda_backend` (decayed-to-scalar).
* `mem_analysis.get_e_mem`: memory of a WindowExpr with `derived` is the derived window's (SpecialWindow).
* `sync_check`: **all** WindowStmt handling deleted (want_value/want_sync/comp_qual_tl `src_buf` branches,
  `comp_index_expr` chaining, WindowStmt case). Only the SpecialWindow-as-proc-parameter fallback to `source_memory_type()` remains (dicey, separate issue).
  Sm90 sync tests (incl. negative ones) and `test_tma` pass.

## Measurements (prototype)

* exo test suite: 1262 tests, all pass except `test_merge_writes_different_lhs_arrays_error`
  (its `find` pattern names a window, which no longer matches).
  Golden changes reviewed: printing of `@ derived(...)`, `2 + i` vs `i + 2` from `chain_window_idx`, one "Required by" comment reorder.
  **No change to generated code otherwise.**
* Scheduled Sm90 gemm (bf16, m2n1, both split-K settings): byte-identical C/CUDA vs today's compiler,
  both with tensorMaps in object code and with tensorMaps inserted by `insert_window_stmt`.
* sporkbench: all 38 examples compile with `exocc`; all 190 generated files (`.c`, `.h`, `.cu`, `.cuh`, excut string table)
  are byte-identical to today's compiler (only `.d` dependency files differ, listing Python source paths).
* `test_BUGGY_WindowStmt.py`: 4/6 now pass (both `inline_window` type bugs, `stage_mem` alias blindness, staging from a plain window).
  The 2 remaining are `DoStageMem` WindowExpr bugs unrelated to WindowStmt.
* Bounds check hole fixed for free: today `w = x[2:12]; w[9] = 1.0` (writes `x[11]`, x has 8 elements) and Reduce through windows are accepted
  (`boundscheck.map_stmts` doesn't translate Assign/Reduce through windows); after desugaring they're rejected.

Instrumented counts (events, i.e. number of desugared/redirected accesses):

| path | exo tests | sporkbench (38 examples) | kind |
|---|---|---|---|
| `desugar_typecheck` (object code indexes a window name) | 131 at 42 lines | 287 at 19 lines | frontend sugar |
| `desugar_inline` (DoInline window args) | 343 | 456 | inherent |
| `inline_window_redirect` | 316 | 456 | inherent |
| `unify` | 71 | 425 | inherent |
| `stage_mem_window_name` (stage_mem string names a window) | 0 | 48 at 1 line (`tk_gemm_util.schedule_gemm`) | **compat** |
| `replace_auto_derive` (special-window search) | 0 | 48 at 2 lines (`schedule_gemm`) | see Open Decisions |

Nearly all sugar traffic is `tk_gemm_util` (handwritten gemm object code + `sched_inline_stuff`) and `gemmini`.

## Open Decisions

1. **`stage_mem` window-name translation** (compat; prototype translates `w[...]` in the staging expression to `x[...]`).
   Only user: `tk_gemm_util.schedule_gemm` (2 calls). Recommendation: delete it and change those call sites to stage `A[...]`/`B[...]`.
2. **How a TMA argument gets `derived` when no matched access had it** (schedule flow: `stage_mem` builds fresh loads with `derived=None`;
   also any flow using `insert_window_stmt`). Options: keep `replace`'s special-window search (unique match or error), or an explicit `set_derived(call, arg, w)` op.
3. **Frontend sugar** (object code `w[i]`): keep (it's the object-language feature; one small pass), or forbid indexing windows in object code
   and add parseable `x[i] @ derived(w)` syntax (would touch ~60 source lines across exo/sporkbench).
4. Window statement's own bounds (window vs its tensor) are unchecked today and in the prototype; matters for tensorMaps (box past the allocation).

## Remaining Cleanup (not attempted)

Branches that only existed to resolve accesses through window names are now dead and could be deleted:
`boundscheck.translate_eff`/window type_env, `new_eff.translate_win` / WindowStmt cases, `mem_analysis.window_alias`,
`LoopIR` free-variable `T.Window.src_buf` handling, `extract_env` special-window case, `cuda_backend`/`coll_analysis` WindowStmt notes.
Also: `insert_window_stmt` doesn't support windows of windows; unify equality is affine-only; printing syntax isn't parseable.
