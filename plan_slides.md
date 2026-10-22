# Plan: Shared Paper/Slides Build (`code_to_tex` package, Beamer)

Status: design agreed with David (2026-10-21). Phases 1-2 (package: spork_style.tex, listing generator, build driver incl. `--arxiv`) implemented 2026-10-21;
phases 3-4 (amspork3 migration, `beamer_nyc25_test` smoke deck) done; phase 5 docs: `spork/docs/code_to_tex/REFERENCE_CARD.md` (terse) and `PRIMER.md` (walkthrough).
Listing filter syntax change and the smoke-test fix list implemented 2026-10-21 (252 tests passing).
Remaining: the arXiv `\colorbox` switch (open question below); phase 6 `--arxiv` exists but is untested on amspork3. See "Implementation notes" at the end.
Where this plan's directive/macro details disagree with the package README / REFERENCE_CARD, the package docs are current.

Depends on: nothing in `exo`.
Requirements source: [plan_slides_human_wishlist.md](plan_slides_human_wishlist.md) (David's wishlist; this plan elaborates it).
First user: `spork/docs/amspork3` (Master's thesis staging) plus slide deck(s) sharing its figures and code.

## Goal

One per-document directory holds a paper (`main.tex`) and any number of Beamer slide decks.
They share style, macros, figures, and code listings, with a single source of truth for anything meaningful:

* Code listings are extracted from `samples.py`, which must pass real `exocc`.
* A figure is one `.tex` file, `\input` by both paper and slides; Beamer overlays animate it on slides,
  and every element states explicitly whether it appears in the (non-animated) paper.
* One code listing source = one generated `.tex` file containing all versions as overlays;
  the paper picks one version.
* Agents can build and *look at* (PNG) a single figure or a single logical slide without building everything.

Non-goals: Typst/other toolchains (rejected: figures must stay shared with LaTeX papers, and models write TikZ far more reliably than CeTZ);
backwards compatibility with the old script (frozen as `spork/docs/legacy_code_to_tex.py`, done 2026-10-21).

## Facts established by experiment (2026-10-20/21, TeX Live on this machine, xelatex)

* `beamer.cls`, `standalone.cls`, `tikz` external lib, `minted` are installed. `overlay-beamer-styles` (aobs-tikz), `latexmk`, `pdfcrop` are **not**.
  `pdftoppm` and ImageMagick `convert` are installed.
* A figure file with `\documentclass[beamer]{standalone}` + `\begin{standaloneframe}` compiles alone to one PDF page per overlay step.
  **Pages are not cropped** (full 128x96mm Beamer page) with xelatex or pdflatex; `[beamer,crop]` fails to compile.
  Previews therefore crop via `convert -trim` (or install `pdfcrop`).
* The same file `\input` into a Beamer frame (main doc loads `\usepackage{standalone}`) works: preamble stripped, overlays live; 1 frame -> N slides.
* The same file `\input` into `article` + `\usepackage{beamerarticle}` + `\usepackage{standalone}`:
  **every overlay spec counts as true** in the paper (so `\node<3->` appears) unless the spec carries a mode part, e.g. `<3| article:0>` (hidden in paper).
* Per-step TikZ style works with `\tikzset{onslide/.code args={<#1>#2}{\only<#1| article:0>{\pgfkeysalso{#2}}}}`;
  paper-only style with `\tikzset{onpaper/.code={\mode<article>{\pgfkeysalso{#1}}}}`.
* Macros defined in a standalone figure's *preamble* vanish when `\input` (preamble is stripped). Put them in shared files or the figure body.
* Beamer has built-in single-frame builds: `\includeonlyframes{a,b}` + `\begin{frame}[label=a]`. No ifdef convention needed.
  Measured on a synthetic 60-frame / 240-page TikZ deck: full build 14s, one frame (4 overlay pages) 0.65s, two frames 0.86s (one xelatex pass each).
  Gotcha: `\includeonlyframes` does not expand its argument; when the label comes from a macro defined on the command line use
  `\ifdefined\sporkOnlyFrame\expandafter\includeonlyframes\expandafter{\sporkOnlyFrame}\fi` in the deck preamble and
  `xelatex -jobname=DECK_frame '\def\sporkOnlyFrame{LABEL}\input{DECK.tex}'` (with a bare `\includeonlyframes{\sporkOnlyFrame}` the build silently produces no pages).
* System python is 3.10 (no `tomllib`); pytest 8.3 is installed.
* `amspork3/amspork3_defs.tex` macros contain many `\hyperref[fig:...]`/`\hyperref[sec:...]` to paper-only labels (e.g. `\sporkTrue`);
  on slides these become undefined references.
* arXiv accepts xelatex (see `~/amspork_arxiv_14930/build_arxiv.sh`, which packages from the `-recorder` `.fls` file list).

## Directory layout (per document; amspork3 is the first instance)

```
spork/docs/
  whitepaper_common.tex, slides_common.tex, colorbox_common.tex   # legacy styles: frozen, used only by old documents
  code_to_tex/                                      # self-contained package (airlift-able out of spork); see "Package" below
    latex/spork_style.tex                           # NEW master style (single source of truth for paper + slides look)
  amspork3.pdf, amspork3_nyc26.pdf                  # viewer copies (evince), written only on successful builds
  amspork3/                                         # = working directory of every build step
    tex_build.json                                  # per-document build config (check command, extra files to copy)
    main.tex                                        # paper (name fixed by convention)
    nyc26.tex                                       # a slide deck (name = --slides argument)
    amspork3_defs.tex                               # document-specific packages + \newcommands (existing convention kept)
    {Name}.tex                                      # sections and figures; label {sec|fig}:{Name} <-> {Name}.tex (existing convention)
    samples.py                                      # all listing source; must pass exocc
    samples/                                        # exocc output        (gitignored)
    code/                                           # code_to_tex output  (gitignored)
    copied_styles/                                  # copies of spork_style.tex etc. (gitignored, chmod a-w)
    preview/                                        # PNGs for agents      (gitignored)
```

* `\input` paths are relative to the document directory (no `amspork3/` prefix).
  The master style is reached as `\input{copied_styles/spork_style.tex}`.
* `copied_styles/` is refreshed on every build (chmod u+w, overwrite, chmod a-w) and is gitignored.
  No hashing/edit detection (would false-positive after `git checkout`); the read-only bit is the only guard.
  Copy set: the package's `latex/*.tex` (`spork_style.tex`, `code_to_tex_support.tex`) plus an optional explicit `extra_styles` list in `tex_build.json`.
  `spork_style.tex` is self-contained (no `\input` of siblings), so no transitive closure and no path rewriting.
  Add-if-needed (not now): `\input` closure over copied files with rewriting to `copied_styles/X`
  (not `TEXINPUTS`: arXiv can't set environment variables, and the directory must build as-is there).
* Only Python-generated/derived stuff is gitignored; everything hand-written is committed.

## LaTeX conventions

### Main files

```latex
% main.tex
\documentclass{article}
\newif\ifsporkSlides \sporkSlidesfalse
\input{copied_styles/spork_style.tex}
\input{amspork3_defs.tex}
...
% nyc26.tex
\documentclass[aspectratio=169]{beamer}
\newif\ifsporkSlides \sporkSlidestrue
\input{copied_styles/spork_style.tex}
\input{amspork3_defs.tex}
...
```

`\ifsporkSlides` is the single distinguishing macro.

### Naming (decided 2026-10-21)

Everything new, and everything carried over from the legacy styles, gets a `spork` prefix (LaTeX macros and colors: `\spork...`, `spork...`; TikZ keys/styles: `spork ...`),
so a reader can tell where an unfamiliar macro came from. Implicit words are dropped for horizontal readability (e.g. no "Box").
This table is authoritative; legacy names elsewhere in this plan describe the *old* files.

| Old / provisional | New | Notes |
|---|---|---|
| `\ifslides` | `\ifsporkSlides` (`\sporkSlidestrue`/`\sporkSlidesfalse`) | |
| `\onlyframe` | `\sporkOnlyFrame` | command-line single-frame build |
| `\reveal`, `\slideOnly`, `\paperOnly` | `\sporkReveal`, `\sporkSlideOnly`, `\sporkPaperOnly` | |
| TikZ `reveal`, `slide only`, `paper only`, `paper style` | `spork reveal`, `spork slide only`, `spork paper only`, `spork paper style` | |
| `\codeListing`, `\codeRow`, `\codeVariant` | `\sporkListing`, `\sporkListingRow`, `\sporkListingVariant` | Row/Variant only appear in generated files |
| `\codeMarkXShift`, `\codeMarkYShift` | `\sporkMarkXShift`, `\sporkMarkYShift` | |
| `\figMaxWidth`, `\titleTag`, `\extraSlides` | `\sporkFigMaxWidth`, `\sporkTitleTag`, `\sporkExtraSlides` | |
| `\sporkPaperLink` | unchanged | |
| `\myKeyA`, `\myKeyB` | `\sporkKeyA`, `\sporkKeyB` | colors `keyColorA/B` -> `sporkKeyColorA/B` |
| `\mainKey`, `\minorKey` | `\sporkMainKey`, `\sporkMinorKey` | colors `mainColor`/`minorColor` -> `sporkMainColor`/`sporkMinorColor` |
| `\magicSection`, `\magicSubsection` | `\sporkSection`, `\sporkSubsection` | section + label, with the gray `sec:fooBar` shown at the right like a line number; for submission builds they may reduce to plain `\section` (label kept, gray text hidden) |
| `\myBiggerTitle`, `\myBiggerTitleExtra` | Beamer frametitle template + `\sporkExtraSlides` | colors `titleColor`/`extraTitleColor` -> `sporkTitleColor`/`sporkExtraTitleColor` |
| `\myTitle`, `\mySub` | **removed** | |
| `\redBox` `\yellowBox` `\greenBox` `\blueBox` `\violetBox` | `\sporkRed` `\sporkYellow` `\sporkGreen` `\sporkBlue` `\sporkViolet` | colors `redBoxFg/Bg` -> `sporkRedFg/Bg`, etc. |
| `\nRedBox` `\nGoldBox` `\nGreenBox` `\nBlueBox` `\nPurpleBox` | `\sporkInvRed` `\sporkInvGold` `\sporkInvGreen` `\sporkInvBlue` `\sporkInvPurple` | "negative": saturated background, light text. Colors `sporkInvRedFg/Bg`, etc. |
| `\codecomment`, `\codehrule` | `\sporkComment`, `\sporkCodeRule` | color `codecommentColor` -> `sporkCommentColor` |
| `\fixminipage` | `\sporkFixMinipage` | |
| TikZ `arrow`, `line` | `spork arrow`, `spork line` | define with `\tikzset`, not deprecated `\tikzstyle` |
| TikZ `normalnode` | `spork normal node` | |
| TikZ `redstyle` ... `violetstyle` | `spork red style` ... `spork violet style` | |
| TikZ `nRedstyle` ... `nPurplestyle` | `spork inv red style` ... `spork inv purple style` | longest names; see note |
| TikZ `rednode` ... `violetnode` | `spork red node` ... `spork violet node` | |
| `\abs` | stays `\abs`, moves to `amspork3_defs.tex` | document-specific math notation, not spork style |
| `keyColor`, `\myKey`, `subColor` | **removed** | see conflict table |

Length note: `spork inv purple style` (22 chars) is the longest name. Since everything in a TikZ option list is a style anyway,
`spork inv purple` / `spork red` would also read fine; decide at implementation (default: keep `style`, as listed).

### Master style: `spork_style.tex` (new; replaces the legacy style files for new documents)

One file, the only source of truth for look, branching on `\ifsporkSlides` internally:

* Generic package loading. Paper: `geometry`, `titlesec`, `beamerarticle`, `hyperref`. Slides: no `geometry` (Beamer conflict); `hyperref` comes from Beamer.
  Both: `fontspec` (FreeSans sans, Ubuntu Mono mono), `xcolor`, `amsmath`/`amssymb`, `standalone`, `tikz` (+ `tikzmark`, `positioning`, `calc`, ...), `\input{copied_styles/code_to_tex_support.tex}` is done by the main file or here (decide at implementation).
* Subsumes `colorbox_common.tex` (`\redBox` ... `\nPurpleBox`, `\codecomment`, `\codehrule`).
* The TikZ color styles (`redstyle` ... `nPurplestyle`; today copy-pasted into `spork_b/main.tex`, `spork_nyc25.tex`, `amspork3.tex`).
* `\tikzset{every picture/.style={font=\sffamily}}` in both modes (figures are sans whatever the body font).
* Slides: sans body (`\familydefault`), David's title/key look, a blank Beamer theme -- no default blue-rectangle style.
* `\sporkPaperLink` and the overlay intent macros (or these go in `code_to_tex_support.tex`; decide at implementation).
* Document-specific packages (e.g. `mathpartir`, `stmaryrd`) and macros stay in the document's `*_defs.tex`.

Start minimal: carry over only what new documents use (inventory below), add the rest on demand.
Legacy `whitepaper_common.tex`, `slides_common.tex`, `colorbox_common.tex` stay untouched for old documents.

**Legacy conflicts, settled by David (2026-10-21).**

| Name | `whitepaper_common.tex` | `slides_common.tex` | `spork_style.tex` |
|---|---|---|---|
| `keyColor`, `\myKey` | RGB 118,185,0 (green) | 170,149,0 (alias of keyColorA) | **removed**; use `sporkKeyColorA` / `\sporkKeyA` |
| `subColor` | 170,149,0 | 179,0,149 | **removed**; superseded by `sporkMainColor` (170,0,170) / `sporkMinorColor` (0,149,232) |
| `titleColor` | black | 179,0,149 (magenta) | `sporkTitleColor`, conditional on `\ifsporkSlides` |
| `\myKeyA` | `keyColorA` | `keyColor` | `\sporkKeyA` uses `sporkKeyColorA` (170,149,0); `\sporkKeyB` uses `sporkKeyColorB` (170,210,0), same in both legacy files |
| `\mySub` | `\hspace{-9mm}` sans | `$\blacktriangleright$` prefix | **removed** (superseded the earlier "conditional" decision) |
| `\myTitle` | `\hspace{-12mm}` + vspace | `\hspace{-22mm}` | **removed**; slides use the Beamer frametitle template |

**Inventory: what amspork3 (paper) uses from the legacy styles.**
`\myKeyA`, `\mainKey`, `\magicSection`, `\magicSubsection`, `\codecomment`, `\codehrule`, `\redBox`, `\abs`, `mainColor`, the TikZ color styles; packages from `whitepaper_common.tex` (`mathpartir`, `stmaryrd`, `mathtools`, `titlesec`, ... -- generic ones to `spork_style.tex`, math-notation ones to `amspork3_defs.tex`).

**Inventory: what `spork_nyc25.tex` (+ its 20 `nyc25_*.tex` figure files) uses** (scanned 2026-10-21; usage counts in parentheses):

* Macros from `slides_common.tex`/`colorbox_common.tex`:
  `\myBiggerTitle` (83; slide title, often `Title \hfill \nBlueBox{GEMM v1}` with a right-aligned tag),
  `\myBiggerTitleExtra` (15; same in `extraTitleColor` 0,149,224, for "Extra Slides" appendix),
  `\myKeyA` (66), `\myKeyB` (61),
  `\blueBox` (55), `\violetBox` (13), `\greenBox` (12), `\redBox` (6), `\yellowBox` (1),
  `\nBlueBox` (10), `\nGoldBox`, `\nGreenBox`, `\nRedBox` (1 each), `\codecomment` (6), `\fixminipage` (2);
  direct colors `blueBoxFg` (20), `greenBoxFg` (12), `redBoxFg` (6), `blueBoxBg` (4).
  -> In Beamer, `\myBiggerTitle`/`\myBiggerTitleExtra` become the frametitle template (title color vs. `extraTitleColor` selected by a frame option or a `\sporkExtraSlides` switch),
  so slides write `\begin{frame}{Title}`; keep a way to put a right-aligned tag in the title (e.g. `\frametitle{Title \hfill \nBlueBox{GEMM v1}}` or a `\sporkTitleTag{...}` macro). No `\hspace{-22mm}` hacks.
* TikZ styles defined inline at the top of `spork_nyc25.tex` and used:
  `arrow` (129; `thick,->,>=stealth`), `line` (12), `normalnode` (33; 3cm x 1cm box), color styles `greenstyle` (31), `nPurplestyle` (23), `violetstyle` (21), `redstyle` (10), `yellowstyle` (7), `bluestyle` (6), `nGreenstyle` (6), `nGoldstyle` (5), `nRedstyle` (3);
  colored node styles `greennode` (10), `violetnode` (7), `rednode` (4), `yellownode`, `bluenode` (1 each);
  diagram-specific `Nnode` (16), `Mnode` (4), `ring` (3).
  -> `spork_style.tex`: `arrow`, `line`, all `*style` color styles (incl. unused `nBluestyle` for completeness), and the `*node` colored variants built on a neutral base node.
  Diagram-specific styles (`Mnode`, `Nnode`, `ring`, ...) belong in the document's `*_defs.tex` or the figure file body, not the master style.
  Never used, don't carry over: `smallnode`, `smallishnode`, `widenode`, `bignode`, `smemnode`, `gmemnode`, `smallishsmemnode`, `producer`, `consumer`, `smallproducer`, `copylatency`, `\consumerBox`, `\producerBox`.
* TikZ libraries: `shapes.geometric`, `arrows` (deprecated; prefer `arrows.meta`, keep `>=stealth` look), `positioning`, `calc`.
* Font sizes: nyc25 wraps nearly everything in `\LARGE` (130), `\large` (70), `\Large` (34), `\huge` (15) because of the 263x148mm page.
  Scaled to Beamer's 160mm width: old 12pt `\LARGE` (20.7pt) ~ 12.6pt, `\huge` (24.9pt) ~ 15pt, `\large` (14.4pt) ~ 8.8pt.
  So a Beamer body size of ~12pt reproduces the old look *without* size wrappers; slide text should not need `\LARGE` everywhere.

Slide page is Beamer 16:9 (160x90mm). This replaces the old "shrink the page so figures look big" trick; scaling is explicit (below).

### Overlay intent macros (every overlay goes through these)

The paper is the final, non-animated figure. Neither "all specs visible" nor "all specs hidden" is a correct default,
so **every element with an overlay spec states its paper intent**; elements without a spec appear in both.

| Intent | Text/content macro | TikZ key | Paper |
|---|---|---|---|
| Revealed on slides, part of the final figure | `\sporkReveal<3->{...}` | `spork reveal=<3->` | shown |
| Slide-only (one-off emphasis, remarks, transient) | `\sporkSlideOnly<3>{...}` | `spork slide only=<3>{keys}` (style), `spork slide only=<3>` (element) | hidden |
| Paper-only | `\sporkPaperOnly{...}` | `spork paper style={keys}`, `spork paper only` | shown/applied |

Implementation: thin wrappers around `\only`/`\uncover` with `| article:0` added for slide-only; TikZ keys via `.code args` as in the experiments.
`reveal` should keep layout space on slides (opacity 0 / `\uncover`) so slides don't jump; TikZ figures additionally use a fixed `\useasboundingbox` when needed.
The build driver lints figure/slide sources: a bare `\only<`, `\uncover<`, `\visible<`, `\onslide<`, `\node<`, `\draw<`, `\path<` (anything `<...>` not via the wrappers) is a warning.
Hard-wired step numbers are accepted for now (no named steps).

### Figures

A figure file is both standalone-compilable and `\input`-able:

```latex
\documentclass[beamer]{standalone}
\newif\ifsporkSlides \sporkSlidestrue
\input{copied_styles/spork_style.tex}
\input{amspork3_defs.tex}
\begin{document}
\begin{standaloneframe}
\begin{tikzpicture} ... \end{tikzpicture}
\end{standaloneframe}
\end{document}
```

Standalone figure compiles use slides mode (`\sporkSlidestrue`; decided 2026-10-21): `[beamer]{standalone}` loads Beamer anyway.
To preview the *paper* rendering of a figure (final figure, slide-only elements hidden), `--figure NAME --paper` compiles a generated wrapper
(`article` + `beamerarticle` + `standalone`, `\sporkSlidesfalse`, `\input{NAME.tex}`) rather than adding a third style mode.

Figure font/look must not depend on the host document's body font (the paper body is serif, slides are sans):
`spork_style.tex` sets `\tikzset{every picture/.style={font=\sffamily}}` in both modes, replacing spork_b's per-figure `{\sffamily ...}` wrappers.
Figures are drawn at paper size. Slides scale explicitly: `\scalebox{1.6}{\input{Fig.tex}}` or `\resizebox{\textwidth}{!}{...}`.
Optional: a figure declares `\sporkFigMaxWidth{...}`; its standalone build warns if the picture is wider (moves most sizing bugs into the fast loop).

### Code listings in documents

* Slides: `\sporkListing{cpu}` inside a frame -> listing version v shows on overlay step v+1.
  Options: `offset=k` (version v on step v+1+k), `reserve` (lines that appear later keep their space: `\visible` instead of `\only`; default is `\only`, i.e. code grows).
* Paper: `\sporkListing[version=4]{cpu}`.
* `prefix=...` option renames tikzmarks when a listing is included twice in one document.

### Hyperlinks

* `\sporkPaperLink{label}{text}`: real `\hyperref` in the paper, plain text on slides.
* `amspork3_defs.tex` internal `\hyperref[...]` must route through the same switch (otherwise slides drown in undefined-reference warnings).
* Option to evaluate later: `xr-hyper` `\externaldocument{main}` in slides so these become working links into `main.pdf`.

## Package: `spork/docs/code_to_tex/`

Everything for this toolchain lives in one self-contained package so it can be airlifted out when Exo-GPU ends:
no imports of `exo`, no hard-wired `spork`/`exocc`/`amspork3` knowledge (those come from the per-document `tex_build.json`), stdlib only (Python 3.10).

```
code_to_tex/
  __init__.py
  __main__.py          # python3 -m code_to_tex samples.py code/      (listing generator)
  listing.py           # directive parsing + rendering
  build.py             # python3 -m code_to_tex.build <dir> ...       (build driver, see below)
  latex/code_to_tex_support.tex  # \sporkListing, \sporkListingRow, \sporkListingVariant, mark shifts, overlay intent macros
  README.md            # user docs (directive reference, LaTeX usage); the reference card can link here
  tests/               # pytest; LaTeX-level tests compile tiny paper + Beamer docs (skipped if xelatex missing)
```

`build.py` copies the package's `latex/*.tex` (master style + listing support) into `copied_styles/`.
Running with `-m` needs the package's parent dir on `sys.path`; `build.py` invokes the generator in-process (import), so only the outer `python3 -m code_to_tex.build` call needs the right cwd/`PYTHONPATH`.

`tex_build.json` (JSON because no `tomllib` in 3.10), e.g. for amspork3:

```json
{
  "check": ["exocc", "samples.py"],
  "listing_sources": ["samples.py"],
  "_comment": "extra_styles: optional files copied into copied_styles/ besides the package's spork_style.tex; must not \\input siblings",
  "extra_styles": [],
  "viewer_copy_dir": ".."
}
```

## Listing generator (`python3 -m code_to_tex`)

Text processor (does not import exo). Usage: `python3 -m code_to_tex samples.py code/`.
Fresh implementation; docs in the package README; pytest suite in the package.

### Directives (prefix `# TeX: ` exactly; apply as in legacy unless stated)

| Directive | Meaning |
|---|---|
| `listing NAME COUNT` | declares listing NAME with versions 0..COUNT-1 (was `version`) |
| `begin FILTERS` / `end FILTERS` | include region; refcounted per (listing, version) as in legacy |
| `remark[!] FILTERS` + next line | extra comment line (bold with `!`) |
| `summary[!]` + next line | keep if easy (legacy semantics: placeholder when outside region) |
| `filbreak` | keep if easy |
| `color FILTERS COLOR "text" [N:I \| N:*]` | color matched text on the next non-directive line |
| `mark FILTERS NAME ["text" [N:I]] [before\|after\|start\|end] [xshift=..] [yshift=..]` | place a tikzmark on the next non-directive line |

Filters as legacy: `*`, `NAME`, `NAME[i]`, half-open `NAME[lo:hi]` (`lo <= v < hi`, like Python slices), and open-ended `NAME[lo:]` / `NAME[:hi]`.

**Color matching.** Text is a shell-style quoted string (parsed with `shlex`). Matches are non-overlapping, left to right, on the raw source line.
* No count given: exactly one match required (error otherwise).
* `N:I`: exactly N matches required; color the I-th (0-based). `N:*`: exactly N matches; color all.
* Several `color` directives may stack on one line. Same character, two different colors -> error.
  Same color twice -> no-op. Adjacent/overlapping same-color spans merge into one box.
* Colors by full name: `red green blue yellow violet nRed nGold nGreen nBlue nPurple` (-> `\<name>Box`), plus `elide` (replace span with `...`, legacy `.`).
* Errors: a `color`/`mark` whose filter matches a version for which the next source line is *not* included (dead directive);
  a `color`/`mark` not followed by a source line; unknown color; unparseable text.
* `color`/`mark` may also target a `remark` line (directive placed before the `remark`).

**Marks.** `\tikzmark{<listing>-<NAME>}` placed before/after the match, or at line start/end.
Vertical placement defaults to a global `\sporkMarkYShift` (and `\sporkMarkXShift`), overridable per mark;
this replaces the hand-tuned `\bugYFix` hack in `~/amspork_arxiv_14930/tex_gpu/00_system_overview.tex`.
Calibrate defaults once with a test page that draws dots at marks (agent checks the PNG).
Marks need >= 2 LaTeX passes; the build driver reruns until stable.

### Outputs per listing

* `code/NAME.tex`: **all versions in one file**. Rows = union of rows across versions in source order;
  each row is `\sporkListingRow{\sporkListingVariant{<version set>}{<tex>}...}` where variants group versions with identical rendering
  (coloring/dedent can differ by version). `\sporkListingVariant` becomes `\only<steps>` (or `\visible` with `reserve`) on slides,
  and a membership test on the selected version in the paper.
* `code/NAME.N.txt`: plain text of version N (unchanged legacy meaning; for humans/agents).
* `code/NAME.N.dbg`: plain text plus formatting, using `⟦blue:accum⟧`, `⟦elide:...⟧`, `⟦@markname⟧`, `⟦remark!:...⟧`.
  Debug output only (not an interchange format). Lets an agent verify highlights without reading TeX.
* Generated TeX keeps the "agents: read the .txt/.dbg instead" header.

Escaping, `$...$`-math-in-comments, comment detection: same as legacy (documented quirks may be fixed).

## Build driver (`python3 -m code_to_tex.build`; no shell=True)

```
python3 -m code_to_tex.build <dir> [--slides DECK]... [--frame LABEL[,LABEL]] [--figure NAME] [--skip-code] [--png] [--where LABEL] [--arxiv OUT.tar.gz]
```

1. Resolve all path arguments to absolute paths, then `os.chdir(<dir>)` for the rest of the run.
2. Refresh `copied_styles/` (see layout).
3. Unless `--skip-code`: run the `check` command from `tex_build.json` (`exocc samples.py`), then the listing generator (strictly before LaTeX). Abort on failure.
   (Possible later: skip automatically when sources are older than outputs.)
4. Build `main.tex` (always, unless `--figure`/`--frame`-only mode), then each `--slides DECK` (`DECK.tex`).
   xelatex `-interaction=nonstopmode -recorder`, stdin `/dev/null`, rerun while the log asks for a rerun or labels/tikzmarks changed (cap ~4 passes).
5. `--frame LABEL[,LABEL]` (requires `--slides DECK`): **the primary agent loop for slide appearance.**
   Builds the real deck (real page bounds, real scaling, real title/margins) restricted to the named frames via `\includeonlyframes`
   (command-line `\def` + `\expandafter` idiom from the facts section), jobname `DECK_frame` so the full deck PDF isn't clobbered, and never touches viewer copies.
   ~0.5-1s per pass vs. 10s+ for a full deck. Typical loop: edit figure -> `--slides nyc26 --frame dag_war --skip-code --png` -> look at the overlay-step PNGs.
   Every deck must start with the `\ifdefined\sporkOnlyFrame...` line (template boilerplate); the build warns if it's missing.
6. `--figure NAME [--paper]`: compile `NAME.tex` standalone only (slides mode: one page per overlay step; `--paper`: the final paper rendering via a generated wrapper). Useful for figure-internal work and paper-size checks; **not** a substitute for `--frame`, since it doesn't show the figure within slide bounds.
7. Warnings, **prominent but non-fatal**: undefined references/citations, overfull/underfull boxes (with file:line), multiply-defined labels, bare overlay specs (lint), figure wider than `\sporkFigMaxWidth`.
   Note Beamer frames can overflow vertically without an overfull warning; the PNG check is the real test.
8. Viewer copies: on a build with no undefined references, copy `main.pdf` -> `../<dir>.pdf` and `DECK.pdf` -> `../<dir>_<DECK>.pdf`
   via temp file + rename (evince crash workaround from `amspork3_tex.sh`). Otherwise leave the old copy and say so.
9. `--png`: render the built pages (all, the selected frame's pages, or the figure's steps) with `pdftoppm`, crop with `convert -trim`, into `preview/`.
10. `--where LABEL`: report which slide pages (and frame labels) contain a given label (frame labels and `\label`s, from the `.aux`), so agents can find "which slides contain the labelled content", then feed the frame label to `--frame`.
11. `--arxiv` (later phase): clean build, then package exactly the `.fls` file list (minus slides, samples, previews, standalone figure PDFs), mark `main.tex` as toplevel (standalone figures also contain `\documentclass`), verify the tarball rebuilds. Port from `~/amspork_arxiv_14930/build_arxiv.sh`.

## Suggested `.gitignore` (per document dir)

```gitignore
# exocc output for samples.py: generated samples.c / samples.h (+ debug/), only kept as proof the listings compile
samples/
# python3 -m code_to_tex output from samples.py: NAME.tex (all versions, overlays), NAME.N.txt (plain), NAME.N.dbg (⟦...⟧ formatting)
code/
# build driver's copies of code_to_tex/latex/*.tex (spork_style.tex, code_to_tex_support.tex) + extra_styles; read-only, edit the originals instead
copied_styles/
# build driver --png renders: cropped page/frame/figure-step PNGs for agents to look at
preview/

# LaTeX auxiliary files (Beamer adds .nav/.snm/.toc/.vrb; -recorder adds .fls; tikzmark positions live in .aux)
*.aux
*.log
*.out
*.fls
*.nav
*.snm
*.toc
*.vrb

# Build outputs: main.pdf, DECK.pdf, --frame jobs, standalone figure PDFs.
# Caution: this also ignores hand-made PDF images; un-ignore those individually (!img/foo.pdf) or keep them in a subdirectory with its own rule.
*.pdf
```

## Phases

1. **`code_to_tex` package: listing generator** + tests + README. Includes the LaTeX side of listings (`\sporkListing`, `\sporkListingRow`, `\sporkListingVariant`, mark shifts) in the package's support `.tex` file that gets copied with the styles.
   Test LaTeX output by compiling tiny paper + Beamer test documents and checking page counts / rendered PNGs.
2. **`code_to_tex.build`** (steps 1-10). Replace `spork/docs/amspork3_tex.sh`.
3. **Migrate amspork3**: move `amspork3.tex` -> `amspork3/main.tex`, `amspork3_samples.py` -> `amspork3/samples.py`; drop `amspork3/` prefixes from `\input`; switch from `whitepaper_common.tex` + inline tikzstyles to `spork_style.tex` (move `mathpartir` etc. into `amspork3_defs.tex`); route `amspork3_defs.tex` hyperrefs through the paper-link switch; add `.gitignore`. Compare page PNGs before/after; intentional style changes from `spork_style.tex` are fine, anything else is a migration bug. Write `spork_style.tex` first (David settles the conflict table), as part of this phase.
4. **First deck**: overlay intent macros, a converted figure sequence (e.g. port the `nyc25_dag*.tex` build-up as the proof figure), lint.
5. **Reference card** (`.md`, requested by David): directory structure, paper/slides LaTeX usage, gitignore.
6. **`--arxiv`** packaging.

## Open questions

* **Listing filter syntax: DECIDED and IMPLEMENTED 2026-10-21.** Requirements (David): filters are distinguished syntactically,
  not by recognizing special words; one directive can apply several colors (vertical space matters for human readers, who will write many slides by hand).
  Filters are always the first word after the directive name, comma-joined, no spaces, for every directive
  (`begin cpu[:4],reorder_loops[0]`, `mark split raw_await end`). This removes `RESERVED_LISTING_NAMES`.
  Color directive grammar:
  ```
  colorDirective → "# TeX: color" filters coloring+
  filters        → filter ("," filter)*          (one word)
  coloring       → colorName match+               (color word is sticky until the next color word)
  match          → quotedString index?
  index          → N ":" I  |  N ":*"            (omitted: exactly 1 match)
  ```
  Example: `# TeX: color cpu[3:] violet "n" 2:0 "seq" blue "k"`. Remaining tokens are classified by shape (quoted / `N:I` / bare word);
  an unknown bare word is an error. Rejected (David optimizes for visual scanning; directives must start where expected):
  `;`-separated clauses with different filters, and trailing `# TeX:` directives on code lines.
  Changing it touches the parser, README, REFERENCE_CARD, test fixtures, and `beamer_nyc25_test/samples.py`.
* **arXiv rejects `\colorbox` highlighting** (auto-rejection by regex), even though the boxes are an accessibility feature
  (e.g. legible for protanopia; the colors were tuned to stay distinguishable). `\sporkRed` etc. mean "our current style's way of
  marking something red", not "a box". Planned: a switch in `spork_style.tex` that renders the highlights without `\colorbox`
  (alternatives TBD: colored text, underline, frame), turned on by `--arxiv`. Deferred by David until slides are written; alternatives untested.

* **Possible future guard (not approved yet): shared-geometry check.** Verified 2026-10-21 that a `spork figure` figure has identical
  node coordinates (to 1e-5pt) standalone, in the paper, and in a deck (before `\scalebox`). What breaks this is host-relative sizing
  (`\small` 26.9pt vs 31.3pt wide, `0.3\linewidth` 147pt vs 129pt), now linted in figure files (`build_log.lint_figure_text`).
  Stronger guard if ever needed: every `spork figure` logs its named nodes' positions/sizes, and the build compares the paper and
  slides logs of the same figure, naming nodes that differ.

## Fix list from the smoke test (2026-10-21; all DONE 2026-10-21)

Implementation choices: `\sporkCodeRule` keeps exact `\hrule` spacing (`\moveright\@totalleftmargin\hbox{\rule{\linewidth}...}`);
fonts via opt-in `spork figure` key + redefinable `\sporkFigFont` (9/11pt), `em`-based TikZ defaults now follow the figure font;
viewer copy names: deck == dir -> `<dir>_slides.pdf`, deck `<dir>_x` -> `<dir>_x.pdf`, else `<dir>_<deck>.pdf`.

1. Viewer copy rule: evince crashes on half-written PDFs and on PDFs with undefined references, so copy only when xelatex produced
   a PDF with **no LaTeX errors and no undefined references** (also refuse on nonzero xelatex exit). Overfull boxes etc. never block (they are what "partial progress" meant).
2. Listing lines wrap silently at colored boxes (`\sporkListingBox` uses breakable `\hspace{-\fboxsep}`): use `\kern`, and set each line in an `\hbox` so overlong lines become Overfull warnings.
3. `\sporkCodeRule` spans to the page edge on slides (running `\hrule` in Beamer's wider vbox): use `\rule{\linewidth}{0.4pt}`.
4. Standalone figure skeleton should be `[beamer,aspectratio=169]{standalone}` (4:3 page clips wide figures in `--figure` previews); update plan, README, `spork_style.tex` header.
5. Figure fonts: use `execute at begin picture={\sffamily}` so node `font=\bfseries` stays sans; consider a shared absolute figure/listing font size (`\sporkFigFont`), since body sizes differ between paper (10/11pt) and slides (12pt).
6. `--figure --png`: crop all steps of a figure to their union bounding box so step-to-step movement is visible.
7. Viewer copy name doubles when deck name == directory name (`beamer_nyc25_test_beamer_nyc25_test.pdf`); e.g. use `<dir>_<deck>.pdf` only when they differ.
8. README tips: `overlayarea` for step-varying text heights; `(pic cs:a -| X)` parses but silently lands at the origin
   (the whole `a -| X` is taken as the mark name), define a `\coordinate` first.

Decided: `\sporkListing` defaults to `\only` (code grows; matches the old per-version-file slides), `reserve` opt-in;
frames holding growing listings are recommended top-aligned (`[t]`), since Beamer's default vertical centering moves *all* lines when the listing grows.
Standalone figures compile in slides mode (see Figures). Tests are pytest inside the package; installing TeX/apt dependencies (`overlay-beamer-styles`, `pdfcrop`, ...) is allowed, choose whichever path is less hassle at implementation time.

## Implementation notes (2026-10-21; the package README and `spork_style.tex` header comments are the detailed reference)

* Package modules: `listing.py` (API) + `listing_parse.py`, `listing_layout.py`, `listing_render.py`; `build.py` + `build_log.py`, `build_aux.py`, `build_arxiv.py`;
  `latex/spork_style.tex`, `latex/code_to_tex_support.tex`. Tests: `cd spork/docs && python3 -m pytest code_to_tex/tests`. No new dependencies were installed.
* **standalone.sty patch**: a figure's own preamble line `\newif\ifsporkSlides` (any `\if...` token) made `\input` of the figure fail ("Incomplete \iffalse")
  because of how standalone skips the figure preamble. `spork_style.tex` redefines standalone's internal `\sa@gobble`; this depends on the installed standalone version.
* **beamerarticle** makes `\textrm` etc. overlay-aware, so unbraced `_\textrm{lo}` inside `\hyperref` arguments stops compiling; write `_{\textrm{lo}}`.
* **Marks inside TikZ node text land in the wrong place under xelatex** (xdvipdfmx moves node text with a PDF transform that `\pdfsavepos` never sees).
  Probably the real cause of the old `\bugYFix` hack. Put listings in normal text/minipages and overlay a `remember picture` picture. Marks use `\pgfmark` (tikzmark library, `pic cs:`).
  Calibrated defaults: `\sporkMarkXShift` = 0pt, `\sporkMarkYShift` = 0.5ex.
* **Beamer pitfall**: in `\begin{frame}{Title}{\small ...}` the braced group becomes the (unshown) frame subtitle, so the content vanishes silently.
* TikZ hiding keys (`spork reveal`, `spork slide only`, `spork paper only`) use opacity 0: layout and anchors are kept, but hidden text stays in the PDF text layer.
  Wrapping a whole element in `\sporkSlideOnly`/`\sporkPaperOnly` removes it entirely.
* Listing lines are one paragraph each (not `\\`-separated), so a listing inside a TikZ node needs `text width` or a minipage.
* `--png` crops only `--figure` renders; deck/frame/paper pages keep page bounds so placement is visible.
* `.gitignore` additionally needs `*_paperpreview.tex`, `*.fls`, `*.bbl`, `*.blg`.
