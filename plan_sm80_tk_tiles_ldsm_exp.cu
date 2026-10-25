// Experiment: does ldmatrix -> mma.sync incur register shuffles (MOV/PRMT)
// for various ways of loading the B operand into ThunderKittens tiles?
//
// V=1: TK row-layout B (N x K, K-major smem), TK warp::load (ldsm4 in order 0,1,2,3), mma_ABt
// V=2: TK col-layout B (K x N, N-major smem), TK warp::load (ldsm4t 0,2,1,3), mma_AB
// V=3: TK row-layout B, custom ldmatrix with permuted lane addresses so PTX
//      outputs are {data[0], data[2], data[1], data[3]}, then mma_ABt
// V=4: like V=1 but A also loaded by custom ldmatrix (sanity check, should equal V=1)
// V=5: TK col-layout A (K x M, M-major smem), TK warp::load (ldsm4t 0,2,1,3), mma_AtB
// V=6: TK col-layout A, custom ldmatrix.trans with permuted lane addresses, outputs in order
//
// Research record for plan_sm80_tk_tiles.md. Build/run (sm_80):
//   F="-std=c++20 --expt-relaxed-constexpr --extended-lambda -diag-suppress 177 -O3 -DKITTENS_AMPERE -I$EXO_KITTENS"
//   nvcc $F -arch=sm_80 -cubin -o x.cubin plan_sm80_tk_tiles_ldsm_exp.cu && cuobjdump -sass x.cubin > x.sass
//   python3 plan_sm80_tk_tiles_shuffle_check.py x.sass
//   nvcc $F -arch=sm_80 -DWITH_MAIN -o x plan_sm80_tk_tiles_ldsm_exp.cu && ./x
#include "kittens.cuh"
using namespace kittens;

constexpr int M = 16, N = 64, K = 64;

template <typename RT, typename ST>
__device__ inline void load_B_rowlayout_permuted(RT& dst, const ST& src) {
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    const int lane = laneid();
    const int mat = lane / 8;          // which of the 4 8x8 matrices this lane addresses
    const int r = lane % 8;            // row within that 8x8 matrix
    const int row_off = (mat / 2) * 8; // matrices 0,1: rows 0-7; matrices 2,3: rows 8-15
    const int col_off = (mat % 2) * 8; // matrices 0,2: cols 0-7; matrices 1,3: cols 8-15
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            int row = i * 16 + row_off + r;
            int col = j * 16 + col_off;
            auto& t = dst.tiles[i][j];
            move<half_2>::ldsm4(t.data[0], t.data[2], t.data[1], t.data[3], src.idx(shared_addr, {row, col}));
        }
    }
}

template <typename RT, typename ST>
__device__ inline void load_A_custom(RT& dst, const ST& src) {
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    const int lane = laneid();
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            int row = i * 16 + lane % 16;
            int col = j * 16 + (lane / 16) * 8;
            auto& t = dst.tiles[i][j];
            move<half_2>::ldsm4(t.data[0], t.data[1], t.data[2], t.data[3], src.idx(shared_addr, {row, col}));
        }
    }
}

template <typename RT, typename ST>
__device__ inline void load_A_collayout_permuted(RT& dst, const ST& src) {
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    const int lane = laneid();
    const int mat = lane / 8, r = lane % 8;
    const int row_off = (mat / 2) * 8, col_off = (mat % 2) * 8;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            int row = i * 16 + row_off + r;
            int col = j * 16 + col_off;
            auto& t = dst.tiles[i][j];
            move<half_2>::ldsm4t(t.data[0], t.data[1], t.data[2], t.data[3], src.idx(shared_addr, {row, col}));
        }
    }
}

template <int V>
__global__ void kern(const half* gA, const half* gB, float* gD, int iters) {
    __shared__ alignas(1024) st_hf<(V >= 5 ? K : M), (V >= 5 ? M : K), false> sA;  // [K][M] for V5/V6
    __shared__ alignas(1024) st_hf<64, 64, false> sB;  // [N][K] for V1/V3/V4, [K][N] for V2
    rt_fl<M, N> d;
    warp::zero(d);
    const int lane = laneid();
    for (int it = 0; it < iters; it++) {
        for (int i = lane; i < M * K; i += 32) ((half*)sA.data)[i] = gA[it * M * K + i];
        for (int i = lane; i < 64 * 64; i += 32) ((half*)sB.data)[i] = gB[it * 64 * 64 + i];
        __syncwarp();
        if constexpr (V >= 5) {
            rt_hf<K, M, ducks::rt_layout::col> a;
            if constexpr (V == 5) warp::load(a, sA);
            else load_A_collayout_permuted(a, sA);
            rt_hf<K, N, ducks::rt_layout::col> b;
            warp::load(b, sB);
            warp::mma_AtB(d, a, b, d);
        } else {
        rt_hf<M, K> a;
        if constexpr (V == 4) load_A_custom(a, sA);
        else warp::load(a, sA);
        if constexpr (V == 1 || V == 4) {
            rt_hf<N, K> b;
            warp::load(b, sB);
            warp::mma_ABt(d, a, b, d);
        } else if constexpr (V == 2) {
            rt_hf<K, N, ducks::rt_layout::col> b;
            warp::load(b, sB);
            warp::mma_AB(d, a, b, d);
        } else {
            rt_hf<N, K> b;
            load_B_rowlayout_permuted(b, sB);
            warp::mma_ABt(d, a, b, d);
        }
        }
        __syncwarp();
    }
    // Dump D; layout doesn't matter for checking correctness between variants
    // that compute the same product, as long as all variants store identically.
#pragma unroll
    for (int j = 0; j < d.width; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            gD[((j * 4 + k) * 32 + lane) * 2 + 0] = d.tiles[0][j].data[k].x;
            gD[((j * 4 + k) * 32 + lane) * 2 + 1] = d.tiles[0][j].data[k].y;
        }
}

template __global__ void kern<1>(const half*, const half*, float*, int);
template __global__ void kern<2>(const half*, const half*, float*, int);
template __global__ void kern<3>(const half*, const half*, float*, int);
template __global__ void kern<4>(const half*, const half*, float*, int);
template __global__ void kern<5>(const half*, const half*, float*, int);
template __global__ void kern<6>(const half*, const half*, float*, int);

#ifdef WITH_MAIN
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
int main() {
    const int iters = 3;
    std::vector<half> A(iters * M * K), Akm(iters * M * K), Bnk(iters * N * K), Bkn(iters * N * K);
    std::vector<float> Aref(iters * M * K), Bref(iters * N * K);
    srand(1);
    for (int i = 0; i < (int)A.size(); i++) { float v = (rand() % 17 - 8) / 8.0f; A[i] = __float2half(v); Aref[i] = v; int it = i / (M * K), m = i % (M * K) / K, k = i % K; Akm[it * M * K + k * M + m] = A[i]; }
    for (int it = 0; it < iters; it++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) {
                float v = (rand() % 17 - 8) / 8.0f;
                Bnk[it * N * K + n * K + k] = __float2half(v);
                Bkn[it * N * K + k * N + n] = __float2half(v);
                Bref[it * N * K + n * K + k] = v;
            }
    half *dAkm, *dA, *dBnk, *dBkn; float* dD;
    cudaMalloc(&dA, A.size() * 2); cudaMalloc(&dAkm, A.size() * 2); cudaMemcpy(dAkm, Akm.data(), A.size() * 2, cudaMemcpyHostToDevice); cudaMalloc(&dBnk, Bnk.size() * 2); cudaMalloc(&dBkn, Bkn.size() * 2);
    cudaMalloc(&dD, M * N * 4);
    cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dBnk, Bnk.data(), Bnk.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dBkn, Bkn.data(), Bkn.size() * 2, cudaMemcpyHostToDevice);
    // Note: TK st tiles are swizzled, and we filled them linearly. So the
    // smem contents are a permutation of the intended tile, identical across variants
    // only if the variants agree on swizzle. Check against a reference that undoes
    // nothing: just compare variants to each other (V1 vs V3 vs V4; V2 separately).
    std::vector<float> out[7];
    for (int V = 1; V <= 6; V++) {
        out[V].resize(M * N);
        const half* B = (V == 2 || V >= 5) ? dBkn : dBnk;
        const half* Ap = V >= 5 ? dAkm : dA;
        if (V == 1) kern<1><<<1, 32>>>(Ap, B, dD, iters);
        if (V == 2) kern<2><<<1, 32>>>(Ap, B, dD, iters);
        if (V == 3) kern<3><<<1, 32>>>(Ap, B, dD, iters);
        if (V == 4) kern<4><<<1, 32>>>(Ap, B, dD, iters);
        if (V == 5) kern<5><<<1, 32>>>(Ap, B, dD, iters);
        if (V == 6) kern<6><<<1, 32>>>(Ap, B, dD, iters);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 1; }
        cudaMemcpy(out[V].data(), dD, M * N * 4, cudaMemcpyDeviceToHost);
    }
    // Reference: D[m][n] = sum_it sum_k A[m][k] * B[n][k]
    // TK row-layout D: tile j, data[k], lane: row = lane/4 + 8*(k%2), col = j*16 + 2*(lane%4) + 8*(k/2)
    for (int V = 1; V <= 6; V++) {
        double maxdiff = 0;
        for (int j = 0; j < N / 16; j++) for (int k = 0; k < 4; k++) for (int lane = 0; lane < 32; lane++) for (int x = 0; x < 2; x++) {
            int m = lane / 4 + 8 * (k % 2), n = j * 16 + 2 * (lane % 4) + 8 * (k / 2) + x;
            double ref = 0;
            for (int it = 0; it < iters; it++) for (int kk = 0; kk < K; kk++)
                ref += Aref[it * M * K + m * K + kk] * Bref[it * N * K + n * K + kk];
            maxdiff = fmax(maxdiff, fabs(out[V][((j * 4 + k) * 32 + lane) * 2 + x] - ref));
        }
        printf("V%d max abs diff vs reference: %g\n", V, maxdiff);
    }
    return 0;
}
#endif
