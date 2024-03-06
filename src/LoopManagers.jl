"""
Module `LoopManagers` provides computing managers to pass to functions using the
performance portability module `ManagedLoops`. It implements the API functions defined by `ManagedLoops` for the provided managers.
Currently supported are SIMD and/or multithreaded execution on the CPU. Offloading to GPU via CUDA and oneAPI is experimental.

Additional iteration/offloading strategies (e.g. cache-friendly iteration) can be implemented by defining
new manager types and implementing specialized versions of [`forall`](@ref) and [`offload`](@ref).
"""
module LoopManagers

using ManagedLoops
import ManagedLoops: offload, to_device, parallel, barrier, master, share

export PlainCPU, MultiThreadCPU, VectorizedCPU, MultiThreadSIMD
export CUDA_GPU, oneAPI_GPU

# Conservative defaults
ManagedLoops.default_manager(::Type{HostManager}) = PlainCPU()

# force specialization on `args...`
const VArgs{N}=Vararg{Any,N}

const Range1 = AbstractUnitRange
const Range2{I,J} = Tuple{I,J}
const Range3{I,J,K} = Tuple{I,J,K}

@inline call_single_index(i, fun, args) = fun((i,), args...)
@inline call_single_index((i,j)::Range2, fun, args) = fun(((i,),(j,)), args...)
@inline call_single_index((i,j,k)::Range3, fun, args) = fun(((i,),(j,),(k,)), args...)

import HostCPUFeatures # for single.jl
import Polyester # for threads.jl

# helper functions
include("julia/check_closure.jl")
include("julia/strict_float.jl")

# CPU managers
include("julia/CPU/simd.jl")
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
    While `KA_GPU` is always available, implementations of [`offload`](@ref)
    are available only if the module `KernelAbstractions` is loaded by the main program or its dependencies.
"""
function KernelAbstractions_GPU end

end # module LoopManagers
