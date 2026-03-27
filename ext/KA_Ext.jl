module KA_Ext

using KernelAbstractions: @kernel, @index, synchronize as KA_sync
using KernelAbstractions.Adapt: adapt, adapt_structure
import KernelAbstractions.Adapt.adapt_storage

using LoopManagers: LoopManagers, Range1, Range2, 
                    KernelAbstractions_GPU, GPUConfig, warpsize, SRange, I32
import ManagedLoops: synchronize, offload

struct KA_GPU{W, B, R, GPU} <: KernelAbstractions_GPU{GPU}
    gpu::GPU
end
KA_GPU{W,B,R}(gpu) where {W,B,R} = KA_GPU{W,B,R,typeof(gpu)}(gpu)

LoopManagers.KernelAbstractions_GPU(gpu) = KA_GPU{warpsize(gpu), 0, 0}(gpu)
LoopManagers.KernelAbstractions_GPU(gpu, ::GPUConfig{nwarp, repeat}) where {nwarp,repeat} = KA_GPU{warpsize(gpu), nwarp, repeat}(gpu)
LoopManagers.configure(mgr::KA_GPU{N}, ::GPUConfig{nwarp, repeat}) where {N,nwarp,repeat} = KA_GPU{N, nwarp, repeat}(mgr.gpu)

adapt_storage(mgr::KA_GPU, x) = adapt_mgr(mgr, x)
adapt_mgr(_, x) = x
adapt_mgr(_, x::AbstractUnitRange) = x
adapt_mgr(mgr, x::AbstractArray) = adapt_storage(mgr.gpu, x)

synchronize(mgr::KA_GPU) = KA_sync(mgr.gpu)
offload(fun, mgr::KA_GPU, irange::Range1, args...) = offload_1D(fun, mgr, irange, args)
offload(fun, mgr::KA_GPU, ijrange::Range2, args...) = offload_2D(fun, mgr, ijrange, args)

#======================= one-dimensional iteration ====================#

function offload_1D(fun::Fun, mgr::KA_GPU{<:Any, 0, 0}, irange, args) where {Fun}
    kernel = KA_1D(mgr.gpu)
    kernel(fun, I32(first(irange) - 1), args; ndrange = length(irange))
    return nothing
end

@kernel function KA_1D(fun, i0, args)
    i = I32(@index(Global, Linear))
    i::UInt32
    @inline fun((i + i0,), args...)
end

function offload_1D(fun::Fun, mgr::KA_GPU{W, B, R}, irange, args) where {Fun, W, B, R}
    block, chunk = W*B, W*B*R
    n = div(length(irange)+(chunk-1), chunk) # split irange into n chunks of size B*R
    kernel = KA_1D_WBR(mgr.gpu, block, (block,n))
    i0, ilast = first(irange), last(irange)
    kernel(fun, Val{(W,B,R)}(), I32(i0), I32(ilast), args)
    return nothing
end

@kernel function KA_1D_WBR(fun, ::Val{WBR}, i0, ilast, args) where WBR
    (W, B, R) = WBR
    chunk = B*R*W
    # i0, ilast are 1-based
    # j indexes chunks of B*R*W indices
    i, j = @index(Global, NTuple)
    i, j = zero_based(i), zero_based(j)
    # each thread takes care of R indices : istart, istart+W*B, ..., istart+(R-1)*W*B
    istart = muladd32(j, Val(chunk), i+i0) # 1-based
    ilast = min(istart+I32((R-1)*W*B), ilast)
    istart::UInt32
    ilast::UInt32
    @inline fun(SRange(istart, W*B, ilast), args...)
end

#==================== two-dimensional iteration ====================#

function offload_2D(fun::Fun, mgr::KA_GPU{<:Any,0,0}, (irange, jrange), args) where {Fun}
    M, N = length(irange), length(jrange)
    i0, j0 = first(irange) - 1, first(jrange) - 1
    kernel = KA_2D(mgr.gpu)
    kernel(fun, I32(i0), I32(j0), args; ndrange = (M, N))
    return nothing
end

@kernel function KA_2D(fun::Fun, i0, j0, args) where {Fun}
    i, j = @index(Global, NTuple)
    ranges = (I32(i) + i0,), (I32(j) + j0,)
    @inline fun(ranges, args...)
end

function offload_2D(fun::Fun, mgr::KA_GPU{W,B,R}, (irange, jrange), args) where {Fun,W,B,R}
    block, chunk = W*B, B*R
    n = div(length(jrange)+(chunk-1), chunk) # split jrange into n chunks of size B*R
    kernel = KA_2D_WBR(mgr.gpu, block, (block,n))
    i0, j0 = first(irange), first(jrange)
    kernel(fun, Val{(W,B,R)}(), I32(i0), I32(last(irange)), I32(j0), I32(last(jrange)), args; ndrange = (block, n))
    return nothing
end

@kernel function KA_2D_WBR(fun, ::Val{WBR}, i0::UInt32, ilast::UInt32, j0::UInt32, jlast::UInt32, args) where WBR
    (W, B, R) = WBR
    chunk = B*R
    # i0, j0, ilast, jlast are 1-based
    # k indexes chunks of B*R columns
    i, k = @index(Global, NTuple)
    i = zero_based(i)
    k = zero_based(k)
    if chunk==1
        istart = i+i0 # 1-based
        jstart = k+j0 # 1-based
        istart::UInt32
        jstart::UInt32
        ranges = SRange(istart, W, ilast), (jstart,)
        @inline fun(ranges, args...)
    else
        # divide block of size B*W, indexed by i, into B warps of size W: i = i1 + j1*W
        j1, i1 = div_rem(i, Val(W))  # 0-based
        # reconstruct jrange: 
        # each warp takes care of R columns : jstart, jstart+B, ..., jstart+(R-1)*B
        jstart = muladd32(k, Val(chunk), j1+j0) # 1-based
        istart = i1+i0 # 1-based
        jstart::UInt32
        istart::UInt32
        irange = SRange(istart, W, ilast)
        jlast = min(jstart+I32((R-1)*B), jlast)
        while jstart <= jlast
            @inline fun((irange, (jstart,)), args...)
            jstart += B
        end
    end
end

#================= index arithmetic =================#

# called from GPU => UInt32
@inline div_rem(n::UInt32, v::Val{N}) where N = n >> logtwo(v), n & I32(N-1)
@inline muladd32(a::UInt32, v::Val, b::UInt32) = (a << logtwo(v))+b

logtwo(::Val{1}) = I32(0)
logtwo(::Val{2}) = I32(1)
logtwo(::Val{4}) = I32(2)
logtwo(::Val{8}) = I32(3)
logtwo(::Val{N}) where N = logtwo(Val(div(N,16)))+UInt32(4)

@inline zero_based(x) = I32(x)-I32(1)

end # module
