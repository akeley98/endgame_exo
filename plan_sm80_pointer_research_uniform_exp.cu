// Which source constructs let ptxas put address math on the uniform datapath (UR registers)?
// A 4-stage smem ring: each k-iteration does cp.async of one 16 B chunk per thread into stage
// (k % 4), barrier, then each warp ldmatrix-loads A at a warp-dependent offset and B, and mma's.
//
// Template knobs:
//   WARP: 0 = threadIdx.x / 32 used directly
//         1 = __shfl_sync broadcast (CUTLASS canonical_warp_idx_sync)
//   STAGE: 0 = (k % 4) * STAGE_BYTES recomputed each iteration (Exo style)
//          1 = stage byte offset carried and incremented with wrap (CUTLASS style)
//   BRANCH: 0 = straight-line
//           1 = loop body under `if (threadIdx.x < 128)` (always true, Exo cuda_threads-style guard)
//           2 = loop body under `if (warp == 0..3)` dispatch, Exo-style warp specialization on threadIdx
//           3 = same dispatch, but branching on the (WARP-selected) `warp` variable
//           4 = same dispatch, branching on blockIdx.x (CTA-uniform)
#include <cstdint>
#include <cuda_fp16.h>

constexpr int STAGE_BYTES = 128 * 64 * 2;  // 128 rows x 64 halves... A and B each 8 KiB per stage

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int WARP, int STAGE, int BRANCH>
__global__ void __launch_bounds__(128) kern(const int4* __restrict__ gA, float* out, int K) {
    extern __shared__ __align__(1024) char smem[];
    const uint32_t sbase = smem_u32(smem);
    const int lane = threadIdx.x % 32;
    int warp;
    if constexpr (WARP == 1) warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    else warp = threadIdx.x / 32;

    // Per-lane ldmatrix offset (row = lane % 16, 16 B chunk = lane / 16), 128 B rows, 128 B swizzle.
    const uint32_t row = lane % 16, chunk = lane / 16;
    const uint32_t lane_off = row * 128 + ((chunk ^ (row % 8)) * 16);
    const uint32_t warp_off = warp * 16 * 128;  // each warp reads its own 16 rows of A

    float d[4] = {0, 0, 0, 0};
    uint32_t stage_off = 0;

    auto body = [&](int k) {
        uint32_t s;
        if constexpr (STAGE == 0) s = (k % 4) * STAGE_BYTES;
        else s = stage_off;
        // cp.async: each thread copies one 16 B chunk of A for this stage.
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sbase + s + threadIdx.x * 16),
                     "l"(gA + k * 128 + threadIdx.x));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "r"(sbase + s + warp_off + lane_off));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                     : "r"(sbase + s + 4096 + lane_off));
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        __syncthreads();
        if constexpr (STAGE == 1) stage_off = (stage_off == 3 * STAGE_BYTES) ? 0 : stage_off + STAGE_BYTES;
    };

    if constexpr (BRANCH == 0) {
        
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
    } else if constexpr (BRANCH == 1) {
        if (threadIdx.x < 128) {
            
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
        }
    } else if constexpr (BRANCH == 3) {
        if (warp < 2) {
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
        } else {
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
        }
    } else if constexpr (BRANCH == 4) {
        if (blockIdx.x < 2) {
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
        } else {
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
        }
    } else {
        // Exo-like: separate code paths selected by threadIdx-derived warp index.
        if (threadIdx.x / 32 < 2) {
            
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
        } else {
            
#pragma unroll 1
            for (int k = 0; k < K; k++) body(k);
        }
    }
    for (int i = 0; i < 4; i++) out[(blockIdx.x * 128 + threadIdx.x) * 4 + i] = d[i];
}

#define INST(W, S, B) template __global__ void kern<W, S, B>(const int4*, float*, int);
INST(0, 0, 0) INST(1, 0, 0) INST(0, 1, 0) INST(1, 1, 0)
INST(0, 0, 1) INST(1, 1, 1) INST(0, 0, 2) INST(1, 1, 2)
INST(0, 1, 3) INST(1, 1, 3) INST(0, 1, 4) INST(1, 1, 4) INST(0, 1, 1) INST(0, 1, 2)
