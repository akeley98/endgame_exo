# Qualitative Timeline Note

For sync stmts, I'm using `Sync_tl` values from Exo's `timelines.py`
which the C++ trace generator has to hard-code as qtl sets.
My pseudocode lacks window annotations;
deduce them by getting tidMap from the `blk_thr` statements
and fill the Qtl by doing whatever Exo does for
TMA (to SMEM or to GMEM, never reduce), wgmma, cp.async, or mma instructions
(implied by context).

# Sm90a ping-pong GEMM, no multicast

    RING = 4
    K = 6
    As = alloc_smem[RING]
    Bs = alloc_smem[RING]
    Cs = alloc_smem[1]
    Dr = alloc_rmem[2]  # One per consumer warpgroup
    mma_bar = alloc_mbarriers[2, infinity](expected_arrive_count=128)
    ep_bar = alloc_mbarriers[2, infinity](expected_arrive_count=32)
    prod_bar = alloc_mbarriers[RING, infinity](expected_arrive_count=32)
    cons_bar = alloc_mbarriers[RING, infinity](expected_arrive_count=128)
    wgmma_gr = alloc_async_group[2]
    tma_gr = alloc_async_group[2]

    # Persistent Kernel Loop; example executes 4 tile tasks only
    for task in range(0, 4):
        ping = task % 2
        m, n = next_task()

        prepare_phase(mma_bar[ping, task // 2])
        prepare_phase(ep_bar[ping, task // 2])

        # Turnstile for consumer warpgroup before entering MMA main loop
        with blk_thr[0:1, 128*ping + 128:128*ping + 256]:
            if task > 0:
                Await(mma_bar[1 - ping, (task - 1) // 2], cuda_in_order, 0)

        # MMA Main Loop
        for k in range(0, K):
            k_cu = K * task + k  # Cumulative k iterations in full kernel lifetime
            ring = k_cu % RING
            phase = k_cu // RING
            prepare_phase(prod_bar[ring, phase])
            prepare_phase(cons_bar[ring, phase])

            with blk_thr[0:1, 0:32]:  # Producer
                if phase > 0:
                    Await(cons_bar[ring, phase - 1], cuda_async_proxy_retired, 0)
                # TMA to SMEM
                (As[ring] := A[m, k]) >> prod_bar[ring, phase]
                (Bs[ring] := B[k, n]) >> prod_bar[ring, phase]
                Arrive(cuda_async_proxy_retired) >> prod_bar[ring, phase]
            with blk_thr[0:1, 128*ping + 128:128*ping + 256]:  # Consumer
                Await(prod_bar[ring, phase], cuda_async_proxy_retired, 0)
                Fence(wgmma_fence_1, wgmma_fence_2)
                if k == 0:
                    # WGMMA
                    Dr[ping] := As[ring] * Bs[ring]
                    Arrive(wgmma_async) >> wgmma_gr[ping]
                else:
                    # WGMMA
                    Dr[ping] += As[ring] * Bs[ring]
                    Arrive(wgmma_async) >> wgmma_gr[ping]
                    Await(wgmma_gr[ping], cuda_generic_and_async_proxy, 1)
                    Arrive(cuda_async_proxy_retired) >> cons_bar[(k_cu-1) % RING, (k_cu-1) // RING]

        with blk_thr[0:1, 128*ping + 128:128*ping + 256]:
            # Wait for all WGMMA and do final cons_bar Arrive.
            k_cu = K * (task + 1) - 1
            Await(wgmma_gr[ping], cuda_generic_and_async_proxy, 0)
            Arrive(cuda_async_proxy_retired) >> cons_bar[k_cu % RING, k_cu // RING]

            # Turnstile for consumer warpgroup after exiting MMA main loop
            Arrive(cuda_in_order) >> mma_bar[ping, task // 2]

            # Epilogue run by consumer warpgroup, uses ep_bar turnstile.
            if task > 0:
                Await(ep_bar[1 - ping, (task - 1) // 2], cuda_in_order, 0)
            Cs[0] := Dr[ping]
            Fence(cuda_in_order, cuda_async_proxy_retired)  # Warpgroup sync
            # TMA to GMEM
            with blk_thr[0:1, 128*ping + 128:128*ping + 160]:  # One warp of consumer
                C[m, n] := Cs[0]
                Arrive(tma_to_gmem_async) >> tma_gr[ping]
                Await(tma_gr[ping], cuda_in_order, 0)
                Arrive(cuda_in_order) >> ep_bar[ping, task // 2]



Note the epilogue warpgroup-scope Fence isn't implemented in Exo,
but the idealized GEMM ignores that.
Scalars stand-in for tiles.
When multiple threads are writing a single scalar
(as in the epilogue), just imagine the pseudocode stands-in
for some thunderkittens op that maps the work across threads.


# Sm90a ping-pong GEMM, multicast

TODO after first example is working

# Sm80 GEMM

TODO after first example is working
