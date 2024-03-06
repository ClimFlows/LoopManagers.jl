module KA_Ext

using KernelAbstractions

using GFBackends: GFBackends, DeviceBackend, Range1, Range2
import GFLoops: to_device, to_host, offload

struct KA_GPU{A, G} <: DeviceBackend
    gpu::G
end
GFBackends.KernelAbstractions_GPU(gpu::G, A) where G = KA_GPU{A,G}(gpu)

to_device(data, ::KA_GPU{A}) where A = A(data)
to_host(data, ::KA_GPU) = Array(data)

@inline function offload(fun, backend::KA_GPU, irange::Range1, args...)
    (; gpu) = backend
    kernel = kernel_KA_1D(gpu, 32)
    kernel(fun, first(irange)-1, args... ; ndrange=length(irange))
    KernelAbstractions.synchronize(gpu)
    return nothing
end

@inline function offload(fun, backend::KA_GPU, (irange, jrange)::Range2, args...)
    (; gpu) = backend
    M, N = length(irange), length(jrange)
    i0, j0 = first(irange)-1, first(jrange)-1
    kernel = kernel_KA_2D(gpu, (32,32), (32,N))
    kernel(fun, i0, j0, last(irange), args... )
    KernelAbstractions.synchronize(gpu)
    return nothing
end

@kernel function kernel_KA_1D(fun, i0, args...)
    i = @index(Global, Linear)
    @inline fun((i+i0,), args...)
end

@kernel function kernel_KA_2D(fun, i0, j0, M, args...)
    i, j = @index(Global, NTuple)
    ranges = (i+i0):32:M, (j+j0,)
    @inline fun(ranges, args...)
end

end
