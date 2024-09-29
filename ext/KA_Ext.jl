module KA_Ext

using KernelAbstractions: @kernel, @index, synchronize as KA_sync
import KernelAbstractions.Adapt.adapt_storage

using LoopManagers: LoopManagers, Range1, Range2
import ManagedLoops: synchronize, offload, DeviceManager

struct KA_GPU{Blocks,G} <: DeviceManager
    gpu::G
    blocks::Val{Blocks}
end
LoopManagers.KernelAbstractions_GPU(gpu, blocks::Tuple{Int,Int} = (0, 0)) =
    KA_GPU(gpu, Val(blocks))

adapt_storage(mgr::KA_GPU, x) = adapt_storage(mgr.gpu, x)
synchronize(mgr::KA_GPU) = KA_sync(mgr.gpu)

#========= one-dimensional iteration ==========#

@inline function offload(fun, backend::KA_GPU, irange::Range1, args...)
    (; gpu) = backend
    kernel = kernel_KA_1D(gpu)
    kernel(fun, first(irange) - 1, args; ndrange = length(irange))
    return nothing
end

@kernel function kernel_KA_1D(fun, i0, args)
    i = @index(Global, Linear)
    @inline fun((i + i0,), args...)
end

#============= two-dimensional iteration =============#

@inline function offload(
    fun::Fun,
    backend::KA_GPU{Blocks},
    (irange, jrange)::Range2,
    args...,
) where {Fun,Blocks}
    (; gpu) = backend
    M, N = length(irange), length(jrange)
    i0, j0 = first(irange) - 1, first(jrange) - 1
    block = Blocks[2] # compile-time constant
    
    if block == 0 # no blocking in inner loop
        kernel = kernel_KA_2D(gpu)
        kernel(fun, i0, j0, args; ndrange = map(length, (irange, jrange)))
    else # inner loop is made of blocks of length `block`
        kernel = kernel_KA_2D_blocked(gpu)
        kernel(
            fun,
            i0,
            Val(block),
            last(irange),
            j0,
            args;
            ndrange = (block, length(jrange)),
        )
    end
    return nothing
end

@kernel function kernel_KA_2D(fun::Fun, i0, j0, args) where {Fun}
    i, j = @index(Global, NTuple)
    ranges = (i + i0,), (j + j0,)
    @inline fun(ranges, args...)
end

@kernel function kernel_KA_2D_blocked(
    fun::Fun,
    i0,
    ::Val{block},
    ilast,
    j0,
    args,
) where {Fun,block}
    i, j = @index(Global, NTuple)
    ranges = (i+i0):block:ilast, (j + j0,)
    @inline fun(ranges, args...)
end

end
