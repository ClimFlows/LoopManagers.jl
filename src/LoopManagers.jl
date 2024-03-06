"""
Module `GFBackends` provides computing backends to pass to functions using the 
performance portability module `GFLoops`. It implements the API functions defined by `GFLoops` for the provided backends.
Currently supported are SIMD and/or multithreaded execution on the CPU. Offloading to GPU via CUDA and oneAPI is experimental. 

Additional iteration/offloading strategies (e.g. cache-friendly iteration) can be implemented by defining 
new backend types and implementing specialized versions of [`forall`](@ref) and [`offload`](@ref).
"""
module GFBackends

using Requires
using GFLoops
import GFLoops: offload, forall, to_host, to_device, parallel, barrier, master, share

import Polyester

export PlainCPU, MultiThreadCPU, VectorizedCPU, MultiThreadSIMD
export CUDA_GPU, oneAPI_GPU

# Conservative defaults
GFLoops.default_backend(::Type{HostBackend}) = PlainCPU()

# force specialization on `args...`
const VArgs{N}=Vararg{Any,N}

const Range1 = AbstractUnitRange
const Range2{I,J} = Tuple{I,J}
const Range3{I,J,K} = Tuple{I,J,K}

@inline call_single_index(i, fun, args) = fun((i,), args...)
@inline call_single_index((i,j)::Range2, fun, args) = fun(((i,),(j,)), args...)
@inline call_single_index((i,j,k)::Range3, fun, args) = fun(((i,),(j,),(k,)), args...)

# helper functions
include("julia/check_closure.jl")
include("julia/strict_float.jl")

# CPU backends
include("julia/CPU/simd.jl")
include("julia/CPU/single.jl")
include("julia/CPU/threads.jl")

# composite backends
include("julia/GPU/fakeGPU.jl")
include("julia/tune.jl")

# GPU backends, active only if CUDA or oneAPI module is loaded somewhere
# we define the types here so that they are exported and documented

"""
    backend = CUDA_GPU()

Returns a backend that offloads computations to CUDA-compatible devices. Array arguments
must be `CuArray`s residing on the device. 
!!! note
    While `CUDA_GPU` is always available, implementations of [`forall`](@ref) and [`offload`](@ref)
    are available only if the module `CUDA` is loaded by the main program or its dependencies.
"""
struct CUDA_GPU <: DeviceBackend end

"""
    backend = oneAPI_GPU()

Returns a backend that offloads computations to oneAPI-compatible devices. Array arguments
must be `oneArray`s residing on the device. 

!!! warning
    At the time of writing this docstring, `oneAPI` is not ready for the show. Expect crashes, memory leaks and dismal performance.

!!! note
    While `oneAPI_GPU` is always available, implementations of [`forall`](@ref) and [`offload`](@ref)
    are available only if the module `oneAPI` is loaded by the main program or its dependencies.
"""
struct oneAPI_GPU <: DeviceBackend end

"""
    backend = KernelAbstractions_GPU(gpu::KernelAbstractions.GPU, ArrayType)
    # examples
    backend = KernelAbstractions_GPU(CUDABackend(), CuArray)
    backend = KernelAbstractions_GPU(ROCBackend(), ROCArray)
    backend = KernelAbstractions_GPU(oneBackend(), oneArray)

Returns a backend that offloads computations to a `KernelAbstractions` GPU backend `gpu`.
Transferring `data` to the device is performed by evaluating `ArrayType(data)`.
!!! note
    While `KA_GPU` is always available, implementations of [`offload`](@ref)
    are available only if the module `KernelAbstractions` is loaded by the main program or its dependencies.
"""
function KernelAbstractions_GPU end

end # module GFBackends
