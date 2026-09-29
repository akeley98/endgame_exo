`syncv_table.cpp` (~3600 lines, "the horror file") implements the sync env of the abstract machine (spork_b "Synchronization Semantics"):
VisRecords (memoized, ref-counted, forwarding), visibility sets as sorted linked lists of `TlSigInterval` (thread range × `QualBitsByVis`),
pending awaits (`HamsterPendingAwaitSet`), barrier state, and the read/mutate/free checks.

Key functions: `alloc_vis_record` (VisRecord creation incl. out-of-order and atomic cases), `union_tl_sig_interval`, `synchronizes_with` (witness),
`any_visible_to` / `all_visible_to` (CheckVisRecord), `AugmentVisRecordCallback` + `from_L2` (augment), `FenceUpdateCommand`, the join-threads command,
`hash_vis_record` (62-bit hash: high 32 bits = max tid of non-atomic-only intervals, used by `hash_bounds_for_arrive` to prune Fence/Arrive work), mutate checks (~line 2255).

The atomic-only intervals span all threads `[0, UINT32_MAX)`; excluding them from the hash's max-tid is what keeps the arrive pruning effective.
This relies on atomic QualTLs never being witnessed (never in L1).

Planned: `QualBitsByVis` → `qual_bits_t`; delete vis flag machinery; SyncEnv told which bits are atomic / out-of-order ([plan_remove_vis_flags.md](../../../../../../plan_remove_vis_flags.md)).
Until then, the Python side (exo `sync_check.py`) shims the no-VF model: it passes the same bits as `L2_full` and `L2_temporal`, never sets `write_only_flag`,
and sets `ooo_flag` iff the initial QualTL is out-of-order class (`exec.cpp` reads `is_ooo` only from that flag, so it can't be dropped yet).
