module KA_Ext

using KernelAbstractions: @kernel, @index, synchronize as KA_sync
import KernelAbstractions.Adapt.adapt_storage

using LoopManagers: LoopManagers, Range1, Range2
import ManagedLoops: synchronize, offload, DeviceManager

struct KA_GPU{A, G} <: DeviceManager
    gpu::G
end
LoopManagers.KernelAbstractions_GPU(gpu::G, A) where G = KA_GPU{A,G}(gpu)

adapt_storage(mgr::KA_GPU, x) = adapt_storage(mgr.gpu, x)
synchronize(mgr::KA_GPU) = KA_sync(mgr.gpu)

@inline function offload(fun, backend::KA_GPU, irange::Range1, args...)
    (; gpu) = backend
    kernel = kernel_KA_1D(gpu)
    kernel(fun, first(irange)-1, args ; ndrange=length(irange))
    return nothing
end

@inline function offload(fun::Fun, backend::KA_GPU, (irange, jrange)::Range2, args...) where Fun
    (; gpu) = backend
    M, N = length(irange), length(jrange)
    i0, j0 = first(irange)-1, first(jrange)-1
#    kernel = kernel_KA_2D(gpu, (32,32), (32,N))
    kernel = kernel_KA_2D(gpu)
    kernel(fun, i0, j0, last(irange), args; ndrange=map(length, (irange,jrange)))
    return nothing
end

@kernel function kernel_KA_1D(fun, i0, args)
    i = @index(Global, Linear)
    @inline fun((i+i0,), args...)
end

@kernel function kernel_KA_2D(fun::Fun, i0, j0, M, args) where Fun
    i, j = @index(Global, NTuple)
#    ranges = (i+i0):32:M, (j+j0,)
    ranges = (i+i0,), (j+j0,)
@inline fun(ranges, args...)
end

end
