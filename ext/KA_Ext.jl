module KA_Ext

using KernelAbstractions: @kernel, @index, synchronize as KA_sync
using KernelAbstractions.Adapt: adapt, adapt_structure
import KernelAbstractions.Adapt.adapt_storage

using LoopManagers: LoopManagers, Range1, Range2
import ManagedLoops: synchronize, offload, DeviceManager

struct KA_GPU{Blocks,G} <: DeviceManager
    gpu::G
    blocks::Val{Blocks}
end
LoopManagers.KernelAbstractions_GPU(gpu, blocks::Tuple{Int,Int} = (0, 0)) =
    KA_GPU(gpu, Val(blocks))

adapt_storage(mgr::KA_GPU, x) = adapt_mgr(mgr, x)
adapt_mgr(_, x) = x
adapt_mgr(_, x::AbstractUnitRange) = x
adapt_mgr(mgr, x::AbstractArray) = adapt_storage(mgr.gpu, x)

synchronize(mgr::KA_GPU) = KA_sync(mgr.gpu)

offload(fun, backend::KA_GPU{Blocks}, irange::Range1, args...) where {Blocks} =
    offload_1D(fun, backend.gpu, Val(Blocks[1]), irange, args)
offload(fun, backend::KA_GPU{Blocks}, ijrange::Range2, args...) where {Blocks} =
    offload_2D(fun, backend.gpu, Val(Blocks[2]), ijrange, args)

#========= one-dimensional iteration ==========#

function offload_1D(fun::Fun, gpu, ::Val{0}, irange, args) where {Fun}
    kernel = kernel_KA_1D(gpu)
    kernel(fun, first(irange) - 1, args; ndrange = length(irange))
    return nothing
end

@kernel function kernel_KA_1D(fun, i0, args)
    i = @index(Global, Linear)
    @inline fun((i + i0,), args...)
end

function offload_1D(fun::Fun, gpu, ::Val{block}, irange, args) where {Fun,block}
    kernel = kernel_KA_1D_blocked(gpu, block, length(irange))
    kernel(fun, first(irange) - 1, Val{block}(), last(irange), args)
    return nothing
end

@kernel function kernel_KA_1D_blocked(fun, i0, ::Val{block}, ilast, args) where {block}
    i = @index(Global, Linear)
    range = (i+i0):block:ilast
    @inline fun(range, args...)
end

#============= two-dimensional iteration =============#

function offload_2D(fun::Fun, gpu, ::Val{0}, (irange, jrange), args) where {Fun}
    M, N = length(irange), length(jrange)
    i0, j0 = first(irange) - 1, first(jrange) - 1
    kernel = kernel_KA_2D(gpu)
    kernel(fun, i0, j0, args; ndrange = (M, N))
    return nothing
end

@kernel function kernel_KA_2D(fun::Fun, i0, j0, args) where {Fun}
    i, j = @index(Global, NTuple)
    ranges = (i + i0,), (j + j0,)
    @inline fun(ranges, args...)
end

function offload_2D(fun::Fun, gpu, ::Val{block}, (irange, jrange), args) where {Fun,block}
    M, N = length(irange), length(jrange)
    i0, j0 = first(irange) - 1, first(jrange) - 1
    kernel = kernel_KA_2D_blocked(gpu, block, (block, N))
    kernel(fun, i0, Val{block}(), last(irange), j0, args; ndrange = (block, N))
    return nothing
end

@kernel function kernel_KA_2D_blocked(fun, i0, ::Val{block}, ilast, j0, args) where {block}
    i, j = @index(Global, NTuple)
    ranges = (i+i0):block:ilast, (j + j0,)
    @inline fun(ranges, args...)
end

end # module
