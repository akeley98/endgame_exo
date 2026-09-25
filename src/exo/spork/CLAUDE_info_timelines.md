Defines the timeline vocabulary used by sync-check and CUDA codegen:

* `Instr_tl`: per-instr timeline (which scopes may call it; with the memory type, selects QualTLs).
* `DeviceScope`: CPU vs CUDA scope; allowed instr-tls; default instr-tl for non-instr accesses.
* `Qual_tl`: qualitative timeline; each gets a bit index (camspork `qual_bits_t` is 32 bits; 19 used as of 2026-09).
* `qual_tl_dict`s (`cuda_rmem_`, `cuda_ram_`, `cuda_tmem_`, `cuda_mbarrier_qual_tl_dict`): `Instr_tl → QualTL | [initial, extended...]`; interpreted by `core/memory.py` (`ext = q`, i.e. the extended set includes the initial QualTL).
* `Sync_tl`: currently a full timeline set plus additional temporal-only members (`_cuda_temporal_quals`); `generate_latex_table` produces the table in `spork_b/gSyncTL.tex`.

Planned changes:

* `Qual_tl` gains an implementation class (in-order / out-of-order / atomic) and a proxy class (generic RAM / async RAM / neither); two atomic QualTLs; `wgmma_rmem_fenced_qual` ([plan_remove_vis_flags.md](../../../plan_remove_vis_flags.md)).
* `Sync_tl` becomes a single set; `cuda_temporal` removed; `cuda_mbarrier_only` and an async-only sync-tl added ([plan_proxy_fence_rules.md](../../../plan_proxy_fence_rules.md)).
