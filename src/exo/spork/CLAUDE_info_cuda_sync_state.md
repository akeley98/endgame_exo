`SyncStateBuilder` lowers each barrier variable (and each `Fence`, which is treated as an anonymous barrier) into C++ member functions of `exo_SyncState` plus inline PTX.
`add_barrier` dispatches on the barrier mechanism:

* `Fence` → `add_wgmma_fence` (`wgmma.fence` only) or `add_garden_variety_or_cluster_sync` (`__syncwarp` / `barrier.cta.sync` / `barrier.cluster.arrive+wait`, then maybe `fence.proxy.async`).
* `CudaMbarrier` → `add_mbarrier_ring`: requires a managed ring buffer dimension; generates `Arrive*`/`Await*` helpers (slot = index % ring_depth, parity = (index / ring_depth) & 1, `pre_arrive` skip branch). TMA instrs call the `Arrive*` helper with `enable = 0` only to get the mbarrier address for `expect_tx`/`complete_tx`.
* `CudaBasicCommitGroup` → `add_commit_group` (`cp.async`, TMA store, wgmma commit/wait groups; never emits a proxy fence).
* `CudaClusterSync` → cluster sync.

Proxy fence decisions are currently subset tests on sync-tls (`cuda_temporal.implements_first(L1)`, `cuda_in_order.implements_second(L2)`, ...);
to be replaced by a single QualTL-proxy-class rule ([plan_proxy_fence_rules.md](../../../plan_proxy_fence_rules.md)).

`add_barrier` used to trigger `CudaDeviceSetupBuilder.require_proxy_fence()` (post-`mbarrier.init` `fence.proxy.async`); removed as folklore in exo `84ef21b4` ([plan_mbarrier_codegen.md](../../../plan_mbarrier_codegen.md)).

`$EXO_STRICT_CLUSTER_MBARRIER` (`strict_cluster_mbarrier_setting()`) selects `.cluster` vs `.cta` scope for mbarriers with cross-CTA arrives ([plan_mbarrier_codegen.md](../../../plan_mbarrier_codegen.md)).
