### CUDA Error Testing Pattern

For testing CUDA error conditions, use the `mkproc` pattern with parameterized positive/negative cases.
**Share as much common code as possible** between positive and negative paths to ensure any error is due to the specific thing being tested, not an unrelated mistake.

**Preferred: Parameterize values used inside a single proc**
```python
def mkproc_feature(num_threads):
    """Test thread count limits - parameterize the value, not the structure"""
    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, num_threads):  # Parameter controls behavior
                    foo = 1.0
    return simplify(test_proc)

def test_feature_positive(compiler):
    compiler.cuda_cpu_test(mkproc_feature, num_threads=32)  # Valid: 32 <= blockDim

def test_feature_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_feature, num_threads=64)  # Invalid: 64 > blockDim
    assert "thread" in str(exc.value).lower()
```

**Also good: Parameterize sync-tl, memory types, or other Exo objects**
```python
def mkproc_fence(first_sync_tl, second_sync_tl):
    """Test Fence sync-tl combinations - objects defined outside @proc work inside"""
    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 32):
                    Fence(first_sync_tl, second_sync_tl)  # Parameters from outside
    return simplify(test_proc)

def test_fence_valid_positive(compiler):
    compiler.cuda_cpu_test(mkproc_fence, first_sync_tl=cuda_in_order, second_sync_tl=cuda_in_order)

def test_fence_invalid_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_fence, first_sync_tl=wgmma_async, second_sync_tl=cuda_in_order)
    assert "sync-tl" in str(exc.value).lower()
```

**Last resort: Different proc structures (only when code paths must differ)**
```python
def mkproc_wgmma_fence(use_warpgroup_unit=True):
    """Only use separate branches when structure must differ"""
    device_fn = CudaDeviceFunction(blockDim=128)  # Share what you can

    if use_warpgroup_unit:
        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with device_fn:
                for task in cuda_tasks(0, 1):
                    for wg in cuda_threads(0, 1, unit=cuda_warpgroup):  # Correct unit
                        Fence(wgmma_fence_1, wgmma_fence_2)
    else:
        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with device_fn:
                for task in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, 128):  # Wrong: individual threads
                        Fence(wgmma_fence_1, wgmma_fence_2)
    return simplify(test_proc)
```

Key points:
- **Maximize shared code** - CudaDeviceFunction, CudaWarps, integers, sync-tl objects, and memory types can be assigned outside `@proc` and used inside
- `mkproc_*` functions return a proc, taking parameters that control valid/invalid code paths
- `compiler.cuda_cpu_test(mkproc_fn, **kwargs)` compiles and optionally runs the proc
- Positive tests use the `golden` fixture to compare generated code against expected output
- Negative tests use `pytest.raises` and assert on specific error message substrings
- Parameters to `mkproc_fn` are passed as kwargs to `cuda_cpu_test`

**What CAN be parameterized in mkproc functions:**
- Integer values for loop ranges: `cuda_threads(0, num_warps, unit=cuda_warp)`
- Integer values for array shapes: `data: f32[num_warps, 32] @ CudaRmem`
- CollUnit objects: `cuda_threads(0, 2, unit=thread_unit)` where `thread_unit` is `cuda_thread` or `cuda_warp`
- CudaDeviceFunction with computed values: `CudaDeviceFunction(blockDim=blockDim)`
- Sync-tl objects, memory types, CudaWarps objects

**Clever parameterization trick - use blockDim to control warp count:**
```python
def mkproc_smem_in_warp_loop(blockDim):
    """blockDim=32: 1 warp, valid. blockDim=128: 4 warps, invalid."""
    device_fn = CudaDeviceFunction(blockDim=blockDim)
    num_warps = blockDim // 32

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                for w in cuda_threads(0, num_warps, unit=cuda_warp):
                    smem: f32[32] @ CudaSmemLinear  # Needs all CTA threads
                    for t in cuda_threads(0, 32):
                        smem[t] = 1.0
    return simplify(test_proc)

# blockDim=32: 1 warp loop iteration, all 32 CTA threads available -> valid
# blockDim=128: 4 warp loop iterations, only 32 of 128 threads available -> error
```

**What CANNOT be parameterized:**
- Index expressions inside subscripts: `arr[cta]` vs `arr[1 - cta]` requires if/else branches
- Loop structure differences: cuda_threads vs seq loop nesting requires if/else branches
- Constants in indices: `arr[w, t]` vs `arr[w, 0]` requires if/else branches

**CudaWarps for controlling thread/warp execution:**
```python
def mkproc_warpgroup_alignment(warp_lo):
    """Use CudaWarps to control which warps execute the operation."""
    device_fn = CudaDeviceFunction(blockDim=256)
    warps = CudaWarps(warp_lo, warp_lo + 4)  # 4 warps = 1 warpgroup

    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with device_fn:
            for task in cuda_tasks(0, 1):
                with warps:  # Use CudaWarps with 'with' statement inside proc
                    for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                        Fence(wgmma_fence_1, wgmma_fence_2)
    return simplify(test_proc)

def test_positive(compiler):
    compiler.cuda_cpu_test(mkproc_warpgroup_alignment, warp_lo=0)  # Aligned: 0 % 4 == 0

def test_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_warpgroup_alignment, warp_lo=1)  # Misaligned: 1 % 4 != 0
    assert "alignment" in str(exc.value).lower()
```

Note: For warpgroup operations (like wgmma fence), warp alignment IS checked at compile time. The `lo` parameter to CudaWarps must be aligned to warpgroup boundaries (multiples of 4 warps).

### CUDA Error Testing Pitfalls

Common mistakes when writing CUDA error tests:

1. **Variables must be used** - Declaring a variable is not enough; dead code elimination removes unused allocations before type/memory checks run. Always read from or write to the variable:
   ```python
   # BAD: packed is never used, type check never runs
   packed: i8[4] @ CudaRmemPacked32

   # GOOD: using the variable triggers the type check
   packed: i8[4] @ CudaRmemPacked32
   packed[0] = 0
   ```

2. **Barriers need distribution dimensions** - Use `barrier[N]` with matching thread tiling to avoid distributed memory errors:
   ```python
   # BAD: barrier not distributed, causes "distributed memory deduction failed"
   bar: barrier @ CudaMbarrier
   for tid in cuda_threads(0, 32):
       Arrive(...) >> bar

   # GOOD: barrier dimension matches thread tiling
   bar: barrier[1] @ CudaMbarrier
   for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
       Arrive(...) >> bar[wg]
   ```

3. **CudaGridConstant is read-only on device** - Can only read from it in device code:
   ```python
   # BAD: writing causes "mutable access" error
   gc: f32[16] @ CudaGridConstant
   gc[0] = 1.0

   # GOOD: read from gc, write to different buffer
   dst[0] = gc[0]
   ```

4. **Proc definitions in parameterized mkproc (last resort)** - If you must have different proc structures (not just different parameter values), define procs inside if/else branches to avoid both being created. **Prefer parameterizing values over branching structures** (see CUDA Error Testing Pattern above):
   ```python
   # BAD: both procs created regardless of parameter
   def mkproc(use_valid=True):
       @proc
       def invalid_proc(): ...  # Always created!
       @proc
       def valid_proc(): ...
       return valid_proc if use_valid else invalid_proc

   # ACCEPTABLE (when structure must differ): only requested proc is created
   def mkproc(use_valid=True):
       device_fn = CudaDeviceFunction(blockDim=128)  # Share what you can!
       if use_valid:
           @proc
           def test_proc(): ...  # Valid version
       else:
           @proc
           def test_proc(): ...  # Invalid version
       return simplify(test_proc)

   # PREFERRED: parameterize values, not structure (see examples above)
   def mkproc(num_threads):
       @proc
       def test_proc(): ...  # Single proc, behavior controlled by parameter
       return simplify(test_proc)
   ```

5. **Test the right error** - Ensure test structure is valid first; structural errors mask the error you're testing for. If you get unexpected errors about distributed memory or bounds, fix the test structure before asserting on error messages.

6. **Permitted index expressions** - Distributed memory indices must be plain variable reads of `cuda_threads` iterators:
   ```python
   # BAD: expression instead of plain variable
   smem[1 - cta, tid] = 1.0  # Error: "Expected single variable name, not 1 - cta"

   # BAD: constant instead of variable
   data[w, 0, i] = 1.0  # Error: "Expected single variable name, not 0"

   # BAD: seq iterator instead of cuda_threads
   for j in seq(0, 2):
       data[j, t, i] = 1.0  # Error: "Expected cuda_threads-loop iterator, not j"

   # GOOD: plain variable reads of cuda_threads iterators
   for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
       for tid in cuda_threads(0, 32):
           smem[cta, tid] = 1.0
   ```

7. **tile_count must match array extent** - Iterator range must match distributed dimension:
   ```python
   # BAD: array has 4 elements but iterator only covers 2
   data: f32[4, 32, 2] @ CudaRmem
   for w in cuda_threads(0, 2, unit=cuda_warp):  # Error: "w.tile_count = 2; must be 4"
       data[w, t, i] = 1.0

   # GOOD: iterator range matches array dimension
   data: f32[4, 32, 2] @ CudaRmem
   for w in cuda_threads(0, 4, unit=cuda_warp):
       data[w, t, i] = 1.0
   ```

### Multi-CTA mbarrier tests and `$EXO_STRICT_CLUSTER_MBARRIER`

Codegen for mbarriers receiving arrives from other CTAs depends on `$EXO_STRICT_CLUSTER_MBARRIER` (see `plan_mbarrier_codegen.md`).
Golden/excut tests of such mbarriers must pin it with `monkeypatch.setenv("EXO_STRICT_CLUSTER_MBARRIER", "0" or "1")`,
otherwise the result depends on the user's environment (and undefined emits a `UserWarning`).
`mkref_mbarriers(..., strict_cluster=True)` generates the matching strict excut reference.
