# Research Record: Sync Semantics vs PTX (for the formal semantics write-up)

Not an implementation plan; the evidence behind
[plan_remove_vis_flags.md](plan_remove_vis_flags.md), [plan_proxy_fence_rules.md](plan_proxy_fence_rules.md), [plan_mbarrier_codegen.md](plan_mbarrier_codegen.md).
Collected 2026-09-23 to 2026-09-25.

Sources (primary unless noted):

* NVIDIA PTX ISA, version 9.4, https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (cited by section number; NVIDIA copyright, never copy into these repos).
* Lustig, Cooksey, Giroux, "Mixed-Proxy Extensions for the NVIDIA PTX Memory Consistency Model", ISCA '22 (https://graymalk.in/papers/isca22.pdf),
  and its Alloy model https://github.com/NVlabs/mixedproxy (`alloy/ptx.als`). NVIDIA-authored, but formalizes PTX 7.5 (pre-cluster, texture/surface/constant proxies only; no async proxy).
* CUTLASS checkout `endgame_exo/cutlass` @ 3f5bafb3 (Jan 2026); ThunderKittens at `~/Downloads/ThunderKittens` @ 6fd51f22 (this is `$EXO_KITTENS`).
* The human's unanswered forum post: https://forums.developer.nvidia.com/t/understanding-the-cta-local-requirements-of-fence-proxy-async-as-documented-by-mixed-proxy-extensions-for-the-nvidia-ptx-memory-consistency-model/358263
* Vulkan spec (secondary for our purposes), "Execution and Memory Dependencies" note.

## The PTX memory model has two layers of different quality

* Core layer: base causality order, synchronizes-with (§8.9.4), release/acquire patterns (§8.8), morally strong (§8.7). Since PTX 6.0; formalized (Lustig et al. ASPLOS '19). Trustworthy.
* Proxy layer: "proxy-preserved base causality order" (§8.9.5). Three bullets: same address generic; same address, same proxy, *same thread block*; aliases with an alias proxy fence on the path.
  **There is no rule for `fence.proxy.async` at all**, so this layer cannot derive even the single-thread generic→async-with-fence pattern.
  The "same thread block" wording dates from the PTX 7.5 texture-cache era (per-SM non-coherent caches).
* The `fence` section says bi-directional proxy fences "take effect within a single thread" and "compose with other forms of synchronization according to the rules of the Memory Consistency Model" — rules that do not exist.
  No CTA-locality warning anywhere in the fence section.

## Findings

### mbarrier scope (core layer)

* `mbarrier.arrive` defaults to `.release.cta`; `try_wait` to `.acquire.cta` (§9.7.15.16.16, .19).
* mbarriers synchronize only via release/acquire patterns whose endpoints are morally strong (§8.9.4); morally strong requires each scope to include the other thread (§8.7).
  So `.cta` gives no happens-before across CTAs. Barrier *counting* (phase completion) is unaffected by scope.
* PTX ISA examples use `.cluster` for cross-CTA mbarriers; libcu++ only offers `.release.cluster`/`.relaxed.cluster` for remote arrives;
  CUTLASS/CuTe use default `.cta` everywhere, no comments (`cutlass/include/cutlass/arch/barrier.h:493`, `sm90_pipeline.hpp:630`).
* CUTLASS likely gets away with it because its cross-CTA arrives only carry WAR; RAW data rides TMA `complete_tx` (`.release.cluster` semantics).
* CUTLASS issue #3643 / PR #3647 (secondary, unconfirmed by NVIDIA): switching remote arrives to CLUSTER fixed B200 mismatches, but the same PR also added a proxy fence and a missing wait — confounded.
* Decision: `EXO_STRICT_CLUSTER_MBARRIER` ([plan_mbarrier_codegen.md](plan_mbarrier_codegen.md)).

### mbarrier init proxy fence is folklore

See `exo/src/exo/spork/mbarrier_init_proxy_fence.md`. TMA accesses its mbarrier operand via the generic proxy (explicit only since PTX 9.3).

### Implicit completion fences

"The completion of a cp{.reduce}.async.bulk operation is followed by an implicit generic-async proxy fence" (§9.7.10.28.2); same sentence for `wgmma.mma_async` (§9.7.17.4).
Phrased as "result made visible", but read as a real fence it also covers async read → generic write (WAR). Not checked for tcgen05.

### Proxy fences and WAR/WAW

* Mixed-proxy paper prose: unsynchronized same-address accesses via different proxies form an "intra-thread data race" (any direction);
  the fence "flushes prior generic accesses and the specified proxy's prior accesses" and invalidates stale caches.
* Alloy model: `proxy_preserved_cause_base` applies to any op pair; `causality` is `irreflexive[optional[com].cause]` with `com = rf + co + fr`;
  WAW via `coherence`. So literally, WAR and WAW across proxies need fences.
  Same-proxy non-generic ops are ordered without fences only within one block (`same_block_r`); fences only affect ops in the same block (`proxy_fence_ops`).
* PTX 9.4 adds scoped uni-directional proxy fences (sm_90+, PTX 8.6+; `::read` variant PTX 9.4):
  `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`,
  `fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster`,
  `fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster`.
  The `clusterlaunchcontrol.try_cancel` example (§9.7.15.18, the instr itself is sm_100) fences a cross-CTA generic *read* → async *write* with release-before-sync (reader CTA) + acquire-after-sync (writer CTA).
  → CUDA is not Vulkan-like for generic→async WAR. Default `nvcc` here is CUDA 12.3 (PTX 8.3), too old for these forms.
* Vulkan: "Write-after-read hazards can be solved with just an execution dependency, but read-after-write and write-after-write hazards need appropriate memory dependencies." The human's recollection of WAW being exempt was wrong.
* Real kernels: TMA+wgmma GEMMs (CUTLASS, ThunderKittens, Exo) never fence async→async WAR, including multicast across CTAs; nobody on the internet is worried.
  ThunderKittens emits `fence.proxy.async.shared::cta` in every `warpgroup::mma_fence` (before each wgmma), which incidentally covers its generic-read → TMA-write cases (backward kernel `l_smem`/`d_smem`).
* Decision: one async proxy per cluster; no fence for async↔async; fence for generic→async ([plan_proxy_fence_rules.md](plan_proxy_fence_rules.md)).

### Atomics across proxies

Morally strong requires "Both operations are performed via the same proxy" (§8.7); atomicity requires morally strong (§8.10.3).
So generic `red`/`atom` vs TMA reduce on the same location is a data race; TMA reduce vs TMA reduce is fine (`.relaxed.gpu`/`.sys` default scope).
Per-CTA/cluster locality only exists in proxy-preserved *ordering*, not proxy identity (Alloy `strong_r` has no block term), so per-cluster async proxies do **not** break cross-cluster TMA reduce atomicity — do not model per-cluster async proxies as distinct proxy tags.
Decision: two atomic QualTLs.

## Open questions / deferred

* tcgen05: where must `fence.proxy.async` go for generic write (CTA 0) → cluster sync → `tcgen05.mma` (CTA 1)? (Question 3 of the forum post; "last" may be too late relative to async-issued tcgen05.)
* Multicast TMA: which CTA "performs" the write for the purposes of §8.9.5 / implicit completion fences?
* Cross-cluster async→async ordering (unmodelled).
* Whether `wgmma.wait_group` retiring implies SMEM reads retired (the human argues it is implied by causality: D depends on A/B).
* Future project: model memory/proxy fences head-on ("on the causality chain") instead of via QualTL set membership.

## Doc/code mismatches found in `spork/docs/spork_b` (not caused by the rewrite)

Fixed by the human in spork `8b719ac`: `AwaitSemantics` (`r.s` → `r.p`), `ArriveSemantics` (`τ_post` → `τ_pre`), `ChecksOnFree` (`cuda_in_order_qual` → `cuda_in_order_ram_qual`).
Remaining: `Witness.tex` "TODO is VF_full needed" (moot after vis flag removal).
