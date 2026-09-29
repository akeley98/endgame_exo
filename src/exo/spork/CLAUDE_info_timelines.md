Defines the timeline vocabulary used by sync-check and CUDA codegen:

* `Instr_tl`: per-instr timeline (which scopes may call it; with the memory type, selects QualTLs).
* `DeviceScope`: CPU vs CUDA scope; allowed instr-tls; default instr-tl for non-instr accesses.
* `Qual_tl`: qualitative timeline; each gets a bit index (camspork `qual_bits_t` is 32 bits; 21 used as of 2026-09-29).
  Each has an implementation class (`in_order` / `out_of_order` / `atomic`) and a proxy class (`generic_ram` / `async_ram` / `neither`); `Qual_tl.get_impl_class_bits` / `get_proxy_class_bits` give masks ([plan_remove_vis_flags.md](../../../plan_remove_vis_flags.md)).
* `InstrQuals(initial, precondition=...)`: value type of `qual_tl_dict`s (`cuda_rmem_`, `cuda_ram_`, `cuda_tmem_`, `cuda_mbarrier_qual_tl_dict`); precondition defaults to `{initial}`, asserts no out-of-order or atomic members. Atomic QualTLs come from the instr's `AtomicityInfo` instead.
* `Sync_tl`: a single timeline set; `implements_first/second` are subset tests; asserts no atomic members.
  `cuda_temporal` is gone; `cuda_mbarrier_only = {mbar}`, `cuda_async_proxy_retired = {mbar, async}` ([plan_proxy_fence_rules.md](../../../plan_proxy_fence_rules.md)).
* `needs_proxy_fence(L1, L2)`: the proxy fence rule, used by `cuda_sync_state.py`.
* `generate_latex_table(table_file, key_file)` writes `spork_b/SyncTLTable.tex` and `spork_b/QualTL.tex` (both `\input` by `spork_b/gSyncTL.tex`); regenerate by hand after changing QualTLs/SyncTLs.
