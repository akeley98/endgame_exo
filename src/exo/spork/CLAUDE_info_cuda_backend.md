This file contains the main implementation for manipulating the `LoopIR` sub-tree under a `CudaDeviceFunction`.
It converts the `LoopIR` loop nest from the form publicly documented by `spork_b` into a form that can be fed to the main `LoopIR → C/C++` compiler to generate the code we want.
The main conversions are:

* Managed ring buffer rewrites
* Translating `cuda_tasks` loops into explicit `blockIdx → index` code
* Translating `cuda_threads` loops into explicit `threadIdx → index` code via the hacky `CodegenPar` loop mode
* Removing distributed dimensions, so the code becomes more like a CUDA program that "programs a single thread"
* Converting the single device task into per-warp-specialization code paths, with unused variables eliminated and `setmaxnreg` injected.

The human is strongly consider converting this from a `LoopIR → LoopIR` rewrite into a `LoopIR → CudaThreadIR` rewrite,
where `CudaThreadIR` is a placeholder name for an IR for doing code manipulations from a more CUDA-like perspective of a program for a single thread.
