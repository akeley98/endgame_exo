// Does swizzle-after-add (Exo PDSL codegen) defeat uniform-register stage offsets?
// SWZ=0: S(lane_addr) + stage   (pre-swizzled per-lane, CUTLASS style)
// SWZ=1: S(lane_addr + stage)   (Exo style; mathematically equal since stage is a multiple of P=1024)
#include <cstdint>
constexpr int STAGE_BYTES = 8192;
__device__ __forceinline__ uint32_t S128(uint32_t a) { return a ^ ((a >> 3) & 0x70); }
template <int SWZ>
__global__ void __launch_bounds__(128) kern(float* out, int K) {
    extern __shared__ __align__(1024) char smem[];
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const uint32_t lane_addr = sbase + warp * 2048 + (lane % 16) * 128 + (lane / 16) * 16;
    uint32_t acc = 0;
#pragma unroll 1
    for (int k = 0; k < K; k++) {
        const uint32_t stage = (k & 3) * STAGE_BYTES;
        const uint32_t addr = SWZ ? S128(lane_addr + stage) : S128(lane_addr) + stage;
        uint32_t a0, a1, a2, a3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
        acc += a0 ^ a1 ^ a2 ^ a3;
        __syncthreads();
    }
    out[blockIdx.x * 128 + threadIdx.x] = acc;
}
template __global__ void kern<0>(float*, int);
template __global__ void kern<1>(float*, int);
