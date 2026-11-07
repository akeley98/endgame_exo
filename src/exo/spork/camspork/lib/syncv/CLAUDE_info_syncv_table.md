`syncv_table.cpp` (~3600 lines, "the horror file") implements the sync env of the abstract machine (spork_b "Synchronization Semantics"):
VisRecords (memoized, ref-counted, forwarding), visibility sets as sorted linked lists of `TlSigInterval` (thread range × `QualBitsByVis`),
pending awaits (`HamsterPendingAwaitSet`), barrier state, and the read/mutate/free checks.

Key functions: `alloc_vis_record` (VisRecord creation incl. out-of-order and atomic cases), `union_tl_sig_interval`, `synchronizes_with` (witness),
`any_visible_to` / `all_visible_to` (CheckVisRecord), `AugmentVisRecordCallback` + `from_L2` (augment), `FenceUpdateCommand`, the join-threads command,
`hash_vis_record` (62-bit hash: high 32 bits = max tid of non-atomic-only intervals, used by `hash_bounds_for_arrive` to prune Fence/Arrive work), mutate checks (~line 2255).

The atomic-only intervals span all threads `[0, UINT32_MAX)`; excluding them from the hash's max-tid is what keeps the arrive pruning effective.
This relies on atomic QualTLs never being witnessed (never in L1).

Done (exo `57d5da92`): `QualBitsByVis` → `qual_bits_t`, vis flag machinery deleted, `syncv_init_t::{ooo_qual_mask, atomic_qual_mask}` ([plan_remove_vis_flags.md](../../../../../../plan_remove_vis_flags.md)).

Debug Inspection interface (end of file) is the read-out path for trace testing; it never touches the hot code.
`debug_get_vis_record_data` and `debug_get_read_vis_record_ids` predate testing; prototype branch `akeley98/camspork_trace_proto` adds
`debug_get_assignment_record_data` (mutate + read VisRecord IDs), `debug_get_barrier_data` (arrive/await counts, flags), and `VisRecordDebugData::flags` ([plan_camspork_test.md](../../../../../../plan_camspork_test.md)).

Deviations from amspork3 found while reading (details in [plan_camspork_test.md](../../../../../../plan_camspork_test.md)):
`on_arrive` uses the home barrier's count for all multicast targets; atomic tl-sigs are added unmasked (a bug: the spec masks them; Exo's `QtlMask` also wrongly lacks atomic qual-tls); `ArriveUpdateCommand` eagerly clears records matching `free_on_arrive` (sets `vis_record_freed_flag`);
JoinThreads (`JoinThreadsUpdateCommand`) has no amspork3 counterpart.
