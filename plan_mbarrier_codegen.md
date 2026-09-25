# Plan: mbarrier Codegen Fixes

Status: part 1 committed (exo `e2409336`); part 2 committed (exo `84ef21b4`).

Independent of the other plans. Research record: [plan_sync_semantics_research.md](plan_sync_semantics_research.md)
and `exo/src/exo/spork/mbarrier_init_proxy_fence.md`.

Code: `exo/src/exo/spork/cuda_sync_state.py` (`add_mbarrier_ring`), `exo/src/exo/spork/cuda_device_setup_builder.py`.

## 1. `EXO_STRICT_CLUSTER_MBARRIER` (done)

Problem: mbarriers that receive arrives from other CTAs of the cluster (`len(cta_xor_list) > 1` in `add_mbarrier_ring`)
use the default `.release.cta` arrive and `.acquire.cta` wait. Under the core (non-proxy) PTX memory model
(§8.9.4 synchronizes-with via release/acquire patterns + §8.7 morally strong), `.cta` scope cannot synchronize with a thread in another CTA.
PTX ISA examples and libcu++ use `.cluster`; CUTLASS/CuTe use `.cta`.
The human's executive decision: match CUTLASS by default (to avoid chasing perf differences vs references), with an opt-in strict mode.

Behavior of `$EXO_STRICT_CLUSTER_MBARRIER` (read at codegen time; `strict_cluster_mbarrier_setting()` in `cuda_sync_state.py`):

* undefined: `.cta` (CUTLASS behavior), and a `UserWarning` for each multi-CTA mbarrier.
* `0`: `.cta`, silently.
* anything else (including empty string): for multi-CTA mbarriers only,
  `mbarrier.arrive.release.cluster.shared::cluster.b64` and `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64`.
  The `test_wait` path (`__CUDA_ARCH__ < 900`) keeps `.cta` since clusters do not exist there.
  Single-CTA mbarriers and the TMA trailing-barrier address helper are unaffected.

Requires PTX 8.0 (`.sem`/`.scope` on mbarrier arrive / try_wait); verified `ptxas` from CUDA 12.3 accepts it for sm_90a.

Tests (`exo/tests/cuda/test_cuda_sync.py`):

* Existing multi-CTA mbarrier excut/golden tests pin `EXO_STRICT_CLUSTER_MBARRIER=0` via `monkeypatch` (goldens unchanged).
* `mkproc_mbarriers` / `mkref_mbarriers` take `strict_cluster`; reference generator emits `.cluster` PTX for multi-CTA barriers when set.
* `test_mbarriers_m4n2d1d2_in_order_to_wgmma_strict_{excut,golden}`, `test_mbarriers_m1n4d2d2_temporal_to_wgmma_strict_excut`.
  The strict golden also compiles with nvcc for sm_90a. The Sm90a excut tests cannot run on the local sm_80 machine (untested at runtime).
* `test_mbarriers_strict_cluster_warning`, `test_mbarriers_strict_cluster_no_warning_1_cta`.

Not covered: the tcgen05 commit path (`generate_arrive` asserts "Luca needs to implement this"), and remote `arrive.expect_tx` (Exo does not emit it).

## 2. Remove the post-init `fence.proxy.async` (done)

`CudaDeviceSetupBuilder.require_proxy_fence()` is triggered in `SyncStateBuilder.add_barrier` when any Arrive/Await sync-tl intersects
`internal_cuda_async_proxy_detection`, and emits `fence.proxy.async` after `mbarrier.init`.
Per PTX 9.3+ (TMA accesses its mbarrier operand via the generic proxy) this is folklore from the CUDA 12.4 Programming Guide; see `exo/src/exo/spork/mbarrier_init_proxy_fence.md`.
The existing init sequence (`barrier.cta.sync` or full `barrier.cluster.arrive/wait`) is sufficient.

Side note: the trigger condition was also wrong (kernels whose TMA-load mbarriers only use `cuda_temporal`/`cuda_in_order` never got the fence;
every existing golden gets it only incidentally through unrelated barriers). Moot once removed.

Work:

* Delete `require_proxy_fence` / `_have_proxy_fence` and the trigger in `add_barrier`; maybe delete `internal_cuda_async_proxy_detection` if unused after [plan_proxy_fence_rules.md](plan_proxy_fence_rules.md).
* `tests/cuda/test_cuda_sync.py`: `MbarrierQualConfig.have_init_proxy_fence` → remove (and its comment "we still need the proxy fence at startup").
* Update goldens with an init fence (all TMA-using goldens; see list via `grep -l fence.proxy.async tests/golden`).

Done as described. Also deleted `_cuda_async_proxy_detection_quals` (only used by `internal_cuda_async_proxy_detection`).
16 goldens updated (only removals of the init fence; two wgmma tests with no mbarriers also lose the now-empty `if (threadIdx.x == 0)` block).
Local results: `tests/cuda` all pass, including `--cuda-run-Sm80`. Sm90a excut/runtime tests (`test_mbarriers_*_to_wgmma_*excut`, `test_tma_Sm90a`, `test_Sm90a_gemm`) need an H100 run.

## Observations (no action planned)

* Per-Arrive `multicast_flags` only produce `// cta_mask:` comments; actual targets come from the per-barrier `cta_xor_list`.
* Arrive count passed to `mbarrier.init` = `arrive_thread_count × len(cta_xor_list)` (every thread arrives; no elect-one).
* The `pre_arrive` branch (`enable = i >= 0`) is the runtime branch the main plan (`CLAUDE.md`, "mbarrier Branch Elimination") wants removed.
