# spork directory

The `spork` directory holds most of the "compiler baked-in" CUDA support code.
There is nevertheless CUDA-specific code in other parts of the codebase:

* `backend/`: shared CPU/CUDA codegen (re-evaluate this decision); glue code for emitting `.cu` and `.cuh` and CUDA utilities
* `core/`: instr/memory base classes; `LoopIR` including CUDA extensions
* `platforms/`: substantial externalized CUDA memory and instruction definitions
* `rewrite/`: some CUDA-specific rewrites

Files mapped so far (TODO map out the rest and generate `CLAUDE_info_*.md` and `CLAUDE_detail_*.md`):

* `cuda_backend.py`: LoopIR → LoopIR lowering of `CudaDeviceFunction` bodies.
* `cuda_sync_state.py`: lowers `Fence`/`Arrive`/`Await` and barrier allocations into `exo_SyncState` C++ member functions and inline PTX.
* `timelines.py`: `Instr_tl`, `DeviceScope`, `Qual_tl`, `Sync_tl` definitions and the per-memory-type `qual_tl_dict`s.
* `sync_check.py`: translates LoopIR into a camspork abstract machine program for `Procedure.sync_check`.
* `camspork/`: C++ abstract machine interpreter (JIT-compiled); the sync env lives in `camspork/lib/syncv/`.
