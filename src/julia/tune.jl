# const Managers = Vector{HostManager}

struct TunedCall
    scores::Vector{Float64}
end
TunedCall(n::Int) = TunedCall(zeros(n))

struct Tune <: ManagedLoops.HostManager
    backends::Vector{HostManager}
    calls::Dict{Any,TunedCall}
end
function Base.show(io::IO, mgr::Tune)
    mgrs = join(("$mgr" for mgr in mgr.backends), ",")
    print(io, "tune([$mgrs])")
end

tune(backends) = Tune(backends, Dict{Any, TunedCall}())

function tune()
    avx(vlen) = LoopManagers.VectorizedCPU(vlen)
    threaded(vlen) = LoopManagers.MultiThread(avx(vlen))
    backends = ([threaded(vlen), avx(vlen)] for vlen in (8, 16, 32))
    return Tune(vcat(backends...), Dict{Any, TunedCall}())
end

# implementation of ManagedLoops API

ManagedLoops.parallel(fun, b::Tune) = fun(b)

function ManagedLoops.offload(fun::Fun, b::Tune, range, args::Vararg{Any,N}) where {Fun<:Function, N}
    (; backends, calls) = b
    sig = (fun, range, signature(args))
    if !(sig in keys(calls))
        calls[sig] = TunedCall(length(backends))
    end
    call = calls[sig]
    picked = pick(call.scores) # index into backends, call.scores
    sample(picked, call.scores, backends[picked], fun, range, args)
    return nothing
end

# signature of function call
signature(x) = typeof(x)
signature(a::AbstractArray) = eltype(a), axes(a)
signature(t::Union{Tuple, NamedTuple}) = map(signature, t)

# pick an index with probability proportional to scores
function pick(scores::Vector{F}) where F
    if minimum(scores)>0
        pick_from(scores)
    else
        # we have not sampled each manager once,
        # => sample among not-yet-sampled managers
        pick_from([score > 0 ? zero(F) : one(F) for score in scores])
    end
end

function pick_from(scores::Vector{F}) where F
    x, y, picked = zero(F), rand()*sum(scores), 1
    for i in eachindex(scores)
        x = x+scores[i]
        x>=y && break
        picked = i+1
    end
    return min(picked, length(scores))
end

function sample(picked, scores, backend, fun::Fun, range, args) where Fun
    compile_elapsedtimes = Base.cumulative_compile_time_ns()
    Base.cumulative_compile_timing(true)
    start = time_ns()
    ManagedLoops.offload(fun, backend, range, args...)
    elapsed = (time_ns()-start)*1e-9
    Base.cumulative_compile_timing(false)
    if Base.cumulative_compile_time_ns() == compile_elapsedtimes # no time spent compiling
        scores[picked] += elapsed^(-0.2)
    else
#        @info "Compilation detected, discarding time sample"
    end
    return nothing
end
