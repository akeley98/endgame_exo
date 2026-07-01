# spork directory

The `spork` directory holds most of the "compiler baked-in" CUDA support code.
There is nevertheless CUDA-specific code in other parts of the codebase:

* `backend/`: shared CPU/CUDA codegen (re-evaluate this decision); glue code for emitting `.cu` and `.cuh` and CUDA utilities
* `core/`: instr/memory base classes; `LoopIR` including CUDA extensions
* `platforms/`: substantial externalized CUDA memory and instruction definitions
* `rewrite/`: some CUDA-specific rewrites

TODO map out files and generate `CLAUDE_info_*.md` and `CLAUDE_detail_*.md`.
