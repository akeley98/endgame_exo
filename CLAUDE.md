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
(or at least not be as humiliated in the results section when I write my master thesis on this project).

The issues break into two major categories.
They are almost independent projects, but I present them together in case there's important common work to be done to redesign the codebase in preparation for the changes.
(In particular, IR changes).

<!--

CIR with PtxCodegen node?

(nvm) InlinePtx magic codegen instr; alternate mode for instrs instead of codegen.

Category 1 Campaign:
    * Sm90_SmemSwizzle window encoder, with both swizzled and unswizzled 32-bit pointer; hope for dead value elimination.

    * Remove "codegen_sync_stmt" hack and replace with explicit LoopIR lowering.
      Only mbarrier makes this hard: launder mbarrier into f64 array that the CudaDeviceSetupBuilder magically initializes.
      mbarrier parity needs to somehow be computed.
      -> simplify() does nothing for this.
      -> IndexRangeEnvironment handles branching later.
      -> solitary barrier testing moves here.

    * inline_codegen for ldmatrix and TMA ... replace with another LoopIR nest, with barrier laundered as f64 and parity given to you magically.

    * Look for Read exprs corresponding to memories with a window encoder. For each:
        * Lift into WindowStmt + 0 indexed.
        * Check for any other WindowStmt in scope that "easily" converts to this one.
          Easily means it's a constant offset that meets the swizzle_period requirement.
          If so, replace the rhs of this WindowStmt with the WindowStmt found.
        * Otherwise, differentiate etc.
        * WindowStmt adjust mode.

    * Do cuda_backend rewrites as before

    * Eliminate branches from range analysis? Need to teach IndexRangeEnvironment about ManagedRingBufferIdx: [0, ring_depth - 1] and eliminate branches.
      Actually need to audit the goldens carefully for this step.

Testing:
    * explicit_gridDim, check task id is as expected.
    * Managed ring buffer testing.
    * Tracing camspork

-->


# Category 1. Inefficient Generated Code

* Major issue: no support for pointer arithmetic / "iterators"
* Minor issue: hard-coded persistent kernel for `cuda_tasks` loops
* Minor issue: mbarrier branch elimination

## Pointer Arithmetic

Exo array indexing is mostly a "one-and-done" thing, where each `a[*i]` gets translated to "window indexing" code, which is typically indexing the C pointer that is the base of the `a` allocation.
Unfortunately, this leads to repetitive, very expensive indexing calculations that compete with FP for GPU time.
We do not support caching this repetitive work except through window statements (`WindowStmt`), which are an extremely limited feature allowing storing an *immutable* view to a sub-tile of a tensor.

What we really need is some way to do pointer arithmetic, incrementing a pointer each loop iteration.
This is really hard because pointer manipulation happens at so many different levels in the Exo externalization machinery, all of which work by gluing C strings together:

* `c_window`: externalized indexing expressions for different memory types.
* `codegen`: each instruction may further alter dereferenced pointers; in particular, the warp-convergent `ldmatrix` instruction offsets each thread's pointer separately.
* `SyncStmt` lowering generating SMEM pointer expressions for `mbarrier`.
* `WindowStmt` as mentioned.

As it stands, the best way forward I see is to rewrite these systems to somehow allow for pointer-like variables in the *backend* of the compiler.
After all steps of `cuda_backend` (converts code to a form ready to translate to CUDA C++, with `cuda_tasks`/`cuda_threads` loops replaced with simpler constructs), we somehow have to do rewrites that have the effect of converting pointer indexing into pointer increment patterns.
(Forget about CPU-only Exo for now).
For example

    for i in seq(0, 10):
        a[foo + 4 * i] = ...

has to lower to something akin to

    int* ptr = &a[foo];
    for (int i = 0; i < 10; ++i) {
        *ptr = ...
        ptr += 4;
    }

A major liability of Exo is avoiding change to the `LoopIR` language.
We can't really adjust LoopIR too much unless we want to change scheduling, and I'm adverse to doing that.
Basically every darn Exo-GPU feature has to "laundered" as something that looks single-threaded enough for the rewrites to understand.
Three major limitations:

* We basically can't have anything resembling a mutable pointer variable in the user-exposed language, as that's too hard to analyze.

* Windows are a heavy-weight abstraction (pointers, even const-only ones, would be preferred)
  but fixing this would require huge changes to instruction unification ...
  and there's been a lot of work laundering complicated constructs like `CUtensorMap` into Exo windows.

* `WindowStmt` doesn't interact well with distributed memory deduction.
  It's currently not possible to declare an "array" of windows that are distributed across threads.
  I didn't add this since this may require difficult new syntax: double indexing `window[selects_window][selects_index]`???

Together this is why all the proposed pointer or window manipulation is done

* after lowering to a form with `cuda_threads` loops eliminated [no more distributed memory], and,

* implicitly instead of explicitly (contrary to Exo's original design goal).

## `cuda_tasks` Improvement

* For `cuda_tasks`, at a minimum there should be a switch between "statically-scheduled persistent kernel" (what is implemented) and "traditional schedule" (1:1 mapping between launched CUDA clusters and Exo-GPU tasks).
  Bonus points for allowing Morton swizzling.
  This potentially interacts heavily with the decision of whether to introduce a new IR or not.

## `mbarrier` Branch Elimination

* `mbarrier` codegen includes a branch to implement the `pre_arrive` feature.
  Would be nice to statically eliminate this when possible, which will entail interaction between multiple levels of compiler features (e.g. range query).
  This branch is not free at runtime; due to the low occupancy of tensor kernels, the SM easily stalls waiting for the branch to resolve, as speculation is not a thing in GPUs.
  Example of when this would be useful is when the first `pipeline_depth`-many iterations of the producer warp are unrolled, so the remaining iterations have the branch around the `Await`-for-consumers eliminated.


# Category 2. Sequential-Parallel Equivalence Ties Your Hands

## `cuda_multi_for` Loop

The `cuda_threads` loop turned out to be inadequate for two reasons:

* Major problem: not expressive enough to implement the warp-specialized H100 ping-pong schedule without inordinate code duplication
* Minor problem: branching on `threadIdx.x` is inefficient for warp-convergent branches, which `ptxas` cannot statically detect.
* Minor problem: can't unroll the "same" loop, in two different warp specialized paths, in different ways.

The sketch of the H100 ping-pong schedule is that there's a single producer warp feeding two consumer warpgroups.
The consumer warpgroups alternate in time.
The producer warp alternates serially between servicing the two warpgroups.

    # Producer
    for ping in seq(0, 2):
        for k in seq(...):
            # ...

    # Consumer
    for ping in cuda_threads(0, 2, unit=cuda_warpgroup):
        # Some synchronization needed to keep the two warpgroups disjoint in time
        for k in seq(...):
            # ...

For sequential parallel equivalence, we have to fuse the `k` loops.
Otherwise, the (unshown) ring buffer will be filled in the wrong order, and the `sync_check` will also complain there's no forward progress guarantee.

    for ping in ???(0, 2):  # uh oh
        for k in seq(...):
            with CudaWarps(name="producer"):
                # ...
            with CudaWarps(name="consumer"):
                # ...

The problem is the `ping` loop doesn't have a consistent loop mode anymore.
It has to be `seq` for the producer path, but `cuda_threads` for the consumer path.
This is the "Y" of my potential XY problem; the "X" I propose is to add some sort of `cuda_multi_for` loop mode that allows specifying

    for ping in cuda_multi_for(0, 2, modes=dict(
        producer=Seq(),
        consumer=CudaThreads(unit=cuda_warpgroup),
    )):
        for k in seq(...):
            with CudaWarps(name="producer"):
                # ...
            with CudaWarps(name="consumer"):
                # ...

I haven't really thought this through in detail or even justified this is the correct solution to the problem.
What I'm imagining for now is:

* The thread mapping function is undefined for child statements of such `cuda_multi_for` loops, except:
* When a `with CudaWarps` block is encountered, the statements in the block have their thread mapping function calculated as-if all parent `cuda_multi_for` loops had the loop mode assigned for the warp variable named in the `CudaWarps` (error if the warp variable is not defined in the `cuda_multi_for`).
* Codegen already runs separately per warp specialization, so lower the `cuda_multi_for` loop based on the behavior of the loop mode assigned to the warp variable for the specialization currently being generated; statically eliminate the loop and its body if the mode for the current warp variable is not defined in the `cuda_multi_for`.
* No code is allowed where the thread mapping function is undefined except for loops and if statements.

TODO: make plans for this.
Testing coverage for this is entirely inadequate.

## `exo_warpid`

Regarding the `threadIdx` divergence issue: for Ampere/Hopper code that is very register-hungry (which all tensor code is pretty much), it's critical that the compiler uses uniform registers (warp-shared) as much as possible.
This seems to be inhibited by `threadIdx` branching, even branching that ostensibly is guaranteed to be warp convergent.
In particular, the following patterns **don't work**:

* Branching on `threadIdx.x / 32` (counterexample, if `blockDim.x % 32 != 0 && blockDim.y >= 2`, then this won't be warp-uniform for threads with `threadIdx.y != 0`.
  Exo-GPU never does this, but `ptxas` doesn't know this.
* Explicit `__syncwarp`.
  Guess what, this doesn't guarantee anything, because a warp can make forward progress as long as all threads reach *some* `__syncwarp`, even if they're not the same statement.

What **will** work: using `__shfl_sync` to broadcast `threadIdx.x / 32` to all threads in the warp at the start of the kernel and storing this in a new `exo_warpid` variable in C++.
This needs to be propagated everywhere in the kernel, and `cuda_threads` and warp specialization code needs to be rewritten as functions of `exo_warpid` instead of `threadIdx.x` whenever possible.

TODO: make plans for this.
Fortunately, testing coverage for this should be at least adequate (`tests/cuda/test_coll_units.py`)

## `seq` Unrolling

We should get the goal of unrolling loops differently as part of `cuda_multi_for`, with different `Seq` loop modes for different warp variables.

Testing: Cook up an `excut` test that includes a C++ line number as part of the payload, to detect loop unrolling.


# Directory Structure

Put `plan_*.md` files here (`endgame_plan`: this file's directory), which can reference other plans as dependencies.

We will mirror the `src` and `tests` directory structure of `exo` here.
While mapping out the `exo` project, add or update these files to the mirror directories:

* `CLAUDE.md`: documents the purpose of the corresponding `exo` directory, and files in the directory (lowest detail).
* `CLAUDE_info_*.md`: documents overall purpose and interface of the corresponding `exo` source file (medium detail).
* `CLAUDE_detail_*.md`: documents implementation details of the corresponding `exo` source file (highest detail).

While thinking about or working on plans, you may have important thoughts on, proposed changes to, or have to track in-progress changes to, `exo` source files.
Add these notes to the associated `CLAUDE*` files, selecting the file based on the level-of-detail of the note.

If these notes are related to implementing the plan, annotate these notes with the associated `plan_*.md` file(s).
If they are just facts about the file / Exo code architecture, don't annotate.
