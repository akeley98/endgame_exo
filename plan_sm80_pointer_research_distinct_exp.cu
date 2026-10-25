// How many uses of a uniform stage offset does ptxas need before it uses UR?
#include <cstdint>
constexpr int STAGE_BYTES = 8192;
template <int NUSES, int DISTINCT>
__global__ void __launch_bounds__(128) kern(float* out, int K) {
    extern __shared__ __align__(1024) char smem[];
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int lane = threadIdx.x % 32;
    uint32_t lane_addr[4];
#pragma unroll
    for (int i = 0; i < 4; i++) lane_addr[i] = sbase + i * 2048 + (lane % 16) * 128 + (((lane / 16) ^ (lane % 8)) * 16) + (DISTINCT ? ((lane * (i + 3)) & 0x70) : 0);
    uint32_t acc = 0;
#pragma unroll 1
    for (int k = 0; k < K; k++) {
        const uint32_t stage = (k & 3) * STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < NUSES; i++) {
            uint32_t a0, a1, a2, a3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(lane_addr[i] + stage));
            acc += a0 ^ a1 ^ a2 ^ a3;
        }
        __syncthreads();
    }
    out[blockIdx.x * 128 + threadIdx.x] = acc;
}
template __global__ void kern<1, 1>(float*, int);
template __global__ void kern<2, 1>(float*, int);
template __global__ void kern<3, 1>(float*, int);
template __global__ void kern<4, 1>(float*, int);
