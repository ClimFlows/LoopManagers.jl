# const Managers = Vector{HostManager}

struct TunedCall
    scores::Vector{Float64}
end
TunedCall(n::Int) = TunedCall(zeros(n))

struct Tune <: ManagedLoops.HostManager
    backends::Vector{HostManager}
    calls::Dict{Any,TunedCall}
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
    mscore = maximum(scores)
    # if all scores are zero, pick a uniformly random index
    mscore>0 || return rand(eachindex(scores))
    # change zero scores (not yet sampled) to maximum score
    scores = [score == 0 ? mscore : score for score in scores]
    # now pick a backend with probability proportional to score
    x, y, picked = zero(mscore), rand()*sum(scores), 1
    for i in eachindex(scores)
        x = x+scores[i]
        x>=y && break
        picked = i+1
    end
    return min(picked, length(scores))
end

function sample(picked, scores, backend, fun::Fun, range, args) where Fun
    start = time_ns()
    ManagedLoops.offload(fun, backend, range, args...)
    elapsed = (time_ns()-start)*1e-9
    scores[picked] += elapsed^(-2)
    return nothing
end
