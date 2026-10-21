Wishlist, not full fleshed-out plan.

# Basic Setup

* Slides and paper (currently, `spork/docs/amspork3`) must share common style and newcommand files,
  except a single macro at the start of the paper and slides main tex file to distinguish the two.
  Style may be conditional (e.g. FreeSans as default font for slides only).
* Keep using `amspork3_defs.tex` for the common commands.
  `spork_b` is old/read-only.
* Code listings are extracted from `.py` files that are passed through real `exocc` (must compile).
  Exception: intentionally-wrong listings can be surrounded by `if False:` or `"""`.
  The point is to auto-check accidental mistakes, so undermining checking is intentionally easy.
* Figures are defined in their own LaTeX file.
  Can be `\input` from either the paper or the slides
  (I am accepting the advice to use `\input` instead of include-graphics).
  May use `only<>` for Beamer (color in details of how this works in the full plan) for "animated" slides.
* Must be possible to specify for each vertex/edge in a figure whether it appears in the sole non-animated paper version
  (so a default of "everything visible" or "everything with `only<>` hidden" is not acceptable.
  Both defaults are wrong in some cases, e.g. the base figure for slides pre-animation may hide some to-be-revealed
  figure components that the paper should show,
  and there may be one-off emphasis or extra remarks for one slide only that doesn't belong in the paper).
* Please retain FreeSans and my color trademarks and don't revert to bog-standard Beamer blue-rectangle style.

# Code-to-tex Functionality

* Freeze the old script and make something new.
  Agentic code will beat the socks off my old throwaway script
  (you have no idea the heart-breaking code quality tradeoffs
  I make because human typing is <<1% of token generation speed).
* Probably stick with the existing version/begin/end directives; define new syntax for directives if preferred.
* Remark is probably useful, summary less-so; keep if easy to retain and test.
* New color system specifies one or more directives that apply to the next non-directive line.
  Specifies color, text to match, and if required, disambiguate which matches to actually color.
  IMPORTANT: For multiple matches, I'd like to require evidence the user/agent knows what's happening.
  The multiple match syntax must specify the number of matches expected (error if not correct)
  and either the 0-based index of the match to color, or a wildcard to color all matches.
  If multiple color directives, error if they contradict (same character colored in two ways).
  BONUS: this system allows full names for colors, not just one character.
* New feature: in same style as the color directive,
  add a way to place a `tikzmark` somewhere in the code.
  See `~/amspork_arxiv_14930/tex_gpu/00_system_overview.tex` (not in repo)
  for example of why I want this,
  near the comment `HACK: the tikzmark/pic-cs coordinates captured above consistently land`
  (also the specific numbers for said HACK need to be parameterizable).
* Actually generate documentation a bunch of tests (cf heartbreaking human typing speed comment).
* Output in preferred form for Beamer (all versions of a single named listing = one tex file/overlay?).
* For paper, must have a way to include a specific version.
  Custom macro is acceptable if needed.
* Outputing non-plain-text is NOT APPROVED.
  If required in light of the anti-ambiguity stance,
  also have a third output format besides tex/plain-txt that includes formatting directives.
  Try to pick chars that won't be confused for C++ or Python,
  but doesn't have to be perfect (just a debug output, not an interchange format).

# Build Functionality

* Directory structure will be some dir holding the common style, figures, code, and main file for paper and slides.
* Will be using `amspork3` dir for the current silde deck but don't hard-wire this.
* Supoprt building paper-alone, or both paper and slides.
* (Slides are ephemeral and take forever to build; I am ok with the long-term bit-rot risk of paper-only builds)
* Agents need to be able to label stuff and quickly figure out which slides contain the labelled content.
* Bonus: for the above use case and to save compilation time,
  if feasible, allow building only a single logical slide
  (all animated versions of a slide).
  I am OK with requiring an ifdef convention for this.
* Prominent WARNING for undefined references and overfull stuff.
  Don't make it a hard error; let me (human) see partial progress.
* Adapt the existing smart build script (evince work-arounds)
  but plz rewrite in Python so I'm not always one typo or shell injection away from deleting my hard drive.
  It coordinates the exocc pass and the code-to-LaTeX translation strictly before the LaTeX compilation.
  BONUS: also add flag for skipping exocc/code-to-latex pass; could be useful in tight loops.

# Other

* I'm not sure what to do about hyperlinks that refer to the paper only as of yet.
  Acceptable for now is just to define a custom `\sporkPaperLink` command
  that compiles to a real hyperlink for the paper but plain text for the slides.
* Generally I'm OK with some repetitive but zero-information boilerplate here-and-there.
  The specific duplication I'm adverse to is copying meaningful information
  (for example, a code listing) that risks going out-of-sync later.
* I can live with the hard-wired numbers for beamer `only<...>` for now.
