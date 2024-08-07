"""
    abstract type SingleCPU<:HostManager end
Parent type for manager executing on a single core. Derived types should specialize
`distribute`[@ref] or `offload_single`[@ref] and leave `offload` as it is.
"""
abstract type SingleCPU<:HostManager end

@inline function offload(fun::Fun, b::SingleCPU, range, args::VArgs{NA}) where {Fun<:Function, NA}
    @inline offload_single(fun, b, range, args, 1,1)
end

@inline function offload_single(fun::Fun, b::SingleCPU, range, args, NT, id) where {Fun<:Function}
    check_boxed_variables(fun)
    drange = distribute(range, b, NT, id)
    @inline fun(drange, args...)
end

"""
    manager = PlainCPU()
Manager for sequential execution on the CPU. LLVM will try to vectorize loops marked with `@simd`.
This works mostly for simple loops and arithmetic computations.
For Julia-side vectorization, especially of mathematical functions, see `VectorizedCPU'.
"""
struct PlainCPU <: SingleCPU end

# Divide work among CPU threads.
@inline distribute(range, ::PlainCPU, NT, id) = distribute_plain(range, NT, id)

@inline function distribute_plain(range::Range1, NT, id)
    start, len = first(range), length(range)
    return (start+div(len*(id-1),NT)):(start-1+div(len*id,NT))
end

# distribute outer (last) range over threads
@inline distribute_plain((ri,rj)::Range2, NT, id) = (ri, distribute_plain(rj,NT,id))
@inline distribute_plain((ri,rj,rk)::Range3, NT, id) = (ri, rj, distribute_plain(rk,NT,id))
@inline distribute_plain((ri,rj,rk,rl)::Range4, NT, id) = (ri, rj, rk, distribute_plain(rl,NT,id))

"""
    manager = VectorizedCPU()

Returns a manager for executing loops with optional explicit SIMD vectorization. Only inner loops
marked with `@vec` will use explicit vectorization. If this causes errors, use `@simd` instead of `@vec`.
Vectorization of loops marked with `@simd` is left to the Julia/LLVM compiler, as with PlainCPU.

!!! note
    `ManagedLoops.no_simd(::VectorizedCPU)` returns a `PlainCPU`.
"""
struct VectorizedCPU{VLen} <: SingleCPU end
no_simd(::VectorizedCPU) = PlainCPU()

const N32 = Int64(HostCPUFeatures.pick_vector_width(Float32))

VectorizedCPU(len=N32) = VectorizedCPU{len}()
Base.show(io::IO, ::VectorizedCPU{VLen}) where VLen= print(io, "VectorizedCPU($VLen)")
"""
Divide work among vectorized CPU threads.
"""
@inline distribute(range, b::VectorizedCPU, NT, id) = distribute_simd(range, b, NT, id)

@inline function _distribute_simd(range::Range1, ::VectorizedCPU{VSize}, NT, id) where VSize
    # this implementation avoids tails except for the last thread
    # but it has a non-identified bug
    start, len = first(range), length(range)
    nvec   = div(len, VSize) # number of vectors that fit entirely in range
    tail   = mod(len, VSize)
    work   = nvec + tail # add tail to estimate and divide work to be done
    vstop  = min(nvec, div(id*work, NT))
    stop   = (id==NT) ? last(range) : (start+VSize*vstop-1)
    vstart = min(nvec, div((id-1)*work, NT))
    start  = start+VSize*vstart
    return VecRange{VSize}(start, stop)
end

@inline function distribute_simd(range::Range1, ::VectorizedCPU{VSize}, NT, id) where VSize
    r = distribute_plain(range, NT, id)
    return VecRange{VSize}(first(r), last(r))
end

# distribute outer (last) range over threads, vectorize inner (first) range
@inline distribute_simd((ri,rj)::Range2, b, NT, id) = (distribute_simd(ri,b,1,1), distribute_plain(rj,NT,id))
@inline distribute_simd((ri,rj,rk)::Range3, b, NT, id) = (distribute_simd(ri,b,1,1), rj, distribute_plain(rk,NT,id))
@inline distribute_simd((ri,rj,rk,rl)::Range4, b, NT, id) = (distribute_simd(ri,b,1,1), rj, rk, distribute_plain(rl,NT,id))

#======================= Vectorized range ====================#

struct VecRange{N} <: AbstractUnitRange{Int}
    start::Int
    vstop::Int # bulk = start:N:vstop (vstop excluded)
    stop::Int  # tail = vstop:stop
    function VecRange{N}(start,stop) where N
        vlen = div(stop+1-start, N)
        vstop = start + N*vlen
        new{N}(start, vstop, stop)
    end
end
# AbstractUnitRange
@inline Base.length(range::VecRange{N}) where N = range.stop-range.start+1
@inline Base.first(range::VecRange) = range.start
@inline Base.last(range::VecRange) = range.stop

# normal / @simd iteration
@inline Base.firstindex(::VecRange) = 0
@inline Base.getindex(range::VecRange, i::Integer) = range.start+i
@inline Base.iterate(range::VecRange) = next_item(range.stop, range.start)
@inline Base.iterate(range::VecRange, prev) = next_item(range.stop, prev+1)
@inline next_item(stop, next) = (next <= stop) ? (next, next) : nothing

# @vec iteration
ManagedLoops.bulk(range::VecRange{N}) where N = VecBulk{N}(range.start, range.vstop)
ManagedLoops.tail(range::VecRange) = range.vstop:range.stop

struct VecBulk{N}
    start::Int
    vstop::Int # bulk = start:N:vstop (vstop excluded)
end

@inline Base.length(range::VecBulk{N}) where N = div(range.vstop-range.start, N)
@inline Base.firstindex(::VecBulk) = 0
@inline Base.getindex(range::VecBulk{N}, i) where N = SIMD.VecRange{N}(range.start+N*i)
@inline Base.iterate(range::VecBulk) = next_bulk(range, range.start)
@inline Base.iterate(range::VecBulk{N}, prev) where N = next_bulk(range, prev+N)
@inline next_bulk(range::VecBulk{N}, next) where N = (next < range.vstop) ? (SIMD.VecRange{N}(next), next) : nothing


# support for @vec if ... else
@inline function ManagedLoops.choose(m::SIMD.Vec{N,Bool}, iftrue, iffalse) where N
    all(m) && return iftrue()
    any(m) || return iffalse()
    return SIMD.vifelse(m, iftrue(), iffalse())
end

# support for managed broadcasting with SIMD
# FIXME: Base.@propagate_inbounds would be safer than @inbounds
@inline function Base.getindex(bc::Broadcast.Broadcasted, i::SIMD.VecRange, J::Vararg{Union{Int, CartesianIndex},N}) where N
    broadcast_getindex_vec(bc, i, CartesianIndex(J))
end

@inline function broadcast_getindex_vec(bc::Broadcast.Broadcasted{<:Any,<:Any,<:Any,<:Any}, i, J)
    args = getargs(bc.args, i, J)
    return bc.f(args...)
end

# recursively constructs (args[1][i,J], args[2][i,J], ...)
@inline getargs(args::Tuple, i, J) = ( getarg(args[1], i, J), getargs(Base.tail(args), i, J)...)
@inline getargs(args::Tuple{Any}, i, J) = ( getarg(args[1], i, J), )
@inline getargs(::Tuple{}, i, J) = ()

# get a single argument at index [i,J]
@inline getarg(A::Union{Ref,AbstractArray{<:Any,0},Number}, _, _) = A[] # Scalar-likes can just ignore all indices
@inline getarg(A, i, J) = @inbounds A[i, CartesianIndex(new_index(A, J))]
@inline getarg(A::AbstractArray, i, J) = @inbounds A[i, new_index(A, J)...]

# truncate indices beyond rank
@inline new_index(A, J::CartesianIndex) = J.I[1:(ndims(A)-1)]
