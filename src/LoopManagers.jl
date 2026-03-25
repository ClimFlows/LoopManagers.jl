"""
Module `LoopManagers` provides computing managers to pass to functions using the
performance portability module `ManagedLoops`. It implements the API functions defined by `ManagedLoops` for the provided managers.
Currently supported are SIMD and/or multithreaded execution on the CPU. Offloading to GPU via CUDA and oneAPI is experimental.

Additional iteration/offloading strategies (e.g. cache-friendly iteration) can be implemented by defining
new manager types and implementing specialized versions of `ManagedLoops.offload`.
"""
module LoopManagers

using ManagedLoops
import ManagedLoops: offload, no_simd, parallel, barrier, master, share

export PlainCPU, MultiThreadCPU, VectorizedCPU, MultiThreadSIMD
export CUDA_GPU, oneAPI_GPU

# Conservative defaults
ManagedLoops.default_manager(::Type{HostManager}) = PlainCPU()

# force specialization on `args...`
const VArgs{N}=Vararg{Any,N}

const Range1 = AbstractUnitRange
const Range2{I,J} = Tuple{I,J}
const Range3{I,J,K} = Tuple{I,J,K}
const Range4{I,J,K,L} = Tuple{I,J,K,L}

@inline call_single_index(i, fun, args) = fun((i,), args...)
@inline call_single_index((i,j)::Range2, fun, args) = fun(((i,),(j,)), args...)
@inline call_single_index((i,j,k)::Range3, fun, args) = fun(((i,),(j,),(k,)), args...)
@inline call_single_index((i,j,k,l)::Range4, fun, args) = fun(((i,),(j,),(k,),(l,)), args...)

import HostCPUFeatures # for single.jl
import SIMD # for single.jl
import Polyester # for threads.jl

# helper functions
include("julia/check_closure.jl")
# include("julia/strict_float.jl")

# CPU managers
# include("julia/CPU/simd.jl")
include("julia/CPU/single.jl")
include("julia/CPU/threads.jl")

# composite managers
include("julia/GPU/fakeGPU.jl")
include("julia/tune.jl")

# KernalAbstractions manager, active only if KA is loaded somewhere
# we define the types here so that they are available and documented
# The implementation is in ext/KA_Ext.jl

"""
    gpu = KernelAbstractions_GPU(gpu::KernelAbstractions.GPU, ArrayType)
    # examples
    gpu = KernelAbstractions_GPU(CUDABackend(), CuArray)
    gpu = KernelAbstractions_GPU(ROCBackend(), ROCArray)
    gpu = KernelAbstractions_GPU(oneBackend(), oneArray)

Returns a manager that offloads computations to a `KernelAbstractions` GPU backend.
The returned manager will call `ArrayType(data)` when it needs
to transfer `data` to the device.
!!! note
    While `KA_GPU` is always available, implementations of [`offload`]
    are available only if the module `KernelAbstractions` is loaded by the main program or its dependencies.
"""
function KernelAbstractions_GPU end

"""
    config = GPUConfig(nwarp, repeat)
Return singleton object `config` influencing how loops are executed on a GPU. Pass `config` to `configure`, and use the resulting manager with `@with` or `offload`. 
- `nwarp` is the number of warps per block ; typical values are 1,2,4,8 ; higher values are beneficial for small kernels, but not for those using many registers
- `repeat` is the number of elements taken care of by each thread ; higher value can help amortize the cost of launching the kernel.
- an excessive value of `nwarp`*`repeat` may result in too few warp blocks to fill the GPU.

For 1D loops: 
- the loop is split into chunks of size `warpsize`*`nwarp`*`repeat` with `warpsize` a GPU-dependent number of threads per warp (usually 32)
- each thread takes care of `repeat` indices separated by `warpsize`*`nwarp`, resulting in
  contiguous memory accesses.
- the number of warp blocks is roughly the loop count divided by `warpsize`*`nwarp`*`repeat`. 

For 2D loops:
- the inner loop is distributed among a single warp, so that data depending only on the outer loop index can be reused.
- the outer loop is distributed among warp blocks.
- each thread takes care of `repeat` outer indices separated by `nwarp`
- the number of warp blocks is roughly the outer loop count divided by `nwarp`*`repeat`. 
"""
struct GPUConfig{NWarp, Repeat} 
    GPUConfig(a,b) = new{Int(a), Int(b)}()
end

"""
    nthreads = warpsize(gpu)
Returns the number of threads per warp for a KernelAbstraction `gpu`. Defaults to 32.
"""
warpsize(::Any) = 32

"""
  range = SRange(start, step, stop)

Return a range similar to `start:step:stop` with the following differences:
- `step` is in the type domain and must be known at compile time
- `start`, `step` and `stop` are converted to `UInt32`
"""
struct SRange{step}
    start::UInt32
    stop::UInt32
    @inline SRange(start, step, stop) = new{I32(step)}(I32(start), I32(stop))
    @inline function SRange{step}(start::UInt32, stop::UInt32) where step
        step::UInt32
        new{step}(start, stop)
    end
end
I32(x) = unsafe_trunc(UInt32, x)

@inline Base.iterate(range::SRange) = next_warp_index(range, range.start)
@inline Base.iterate(range::SRange{step}, prev) where step = next_warp_index(range, I32(prev+step))
@inline next_warp_index(range::SRange{step}, next) where step = (next <= range.stop) ? (next, next) : nothing

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end # module LoopManagers
