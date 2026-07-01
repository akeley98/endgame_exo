# Introduction

We have a very difficult task ahead of us.
The hope with the Exo-GPU programming language was that,
for tensor-y programs that can tolerate the no-data-dependent-control-flow and affine-indexing restrictions of Exo,
that Exo-GPU would deliver C-like control with a high degree of safety checking for program rewrites and parallelism.
Despite good intentions, we failed to truly deliver on the ``C-like control'' aspect,
and Exo-GPU-generated programs suffer from a lot of cumulative little inefficiencies not present in competently-written CUDA C++.

Honestly, the underlying issue is that core design choices of Exo-GPU maybe were not great abstractions for the GPU, in hindsight.
Nevertheless, I'd like to make a valiant final effort to fix as much as we can.
Then learn from this experience to design better programming languages for my next project
(or at least not be as humiliated in the results section when I write my master thesis on this project)

# Directory Structure

Put `plan_*.md` files here (`endgame_plan`: this file's directory)

We will mirror the `src` and `tests` directory structure of `exo` here.
While mapping out the `exo` project
In these directories:

* `CLAUDE.md`: documents the purpose of the corresponding `exo` directory, and files in the directory (lowest detail).
* `CLAUDE_info_*.md`: documents overall purpose and interface of the corresponding `exo` source file (medium detail).
* `CLAUDE_detail_*.md`: documents implementation details of the corresponding `exo` source file (highest detail).

While thinking about or working on plans, you may have important thoughts on, proposed changes to, or have to track in-progress changes to, `exo` source files.
Add these notes to the associated `CLAUDE*` files, selecting the file based on the level-of-detail of the note.
Annotate these notes with the associated `plan_*.md` file(s).
