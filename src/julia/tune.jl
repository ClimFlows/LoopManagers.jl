# const Managers = Vector{HostManager}

# Log-normal distributions

struct LogNormal
    N::Int
    mean::Float64
    var::Float64
end
LogNormal()=LogNormal(0,0,0)

function draw(law::LogNormal)
    law.N<2 && return -Inf
    return randn()*sqrt(law.var)+law.mean
end

function push(law::LogNormal, t)
    (;N, var, mean) = law
    logt = log10(t)
    mean_new = (N*mean + logt)/(N+1)
    var_new = (N*var + (logt-mean)*(logt-mean_new))/(N+1)
    return LogNormal(N+1, mean_new, var_new)
end

# Statistics of calling a certain function signature
struct TunedCall
    stats::Vector{LogNormal}
end
TunedCall(n::Int) = TunedCall([LogNormal() for i=1:n])

# Auto-tuning manager
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
    stats = call.stats
    picked = pick(stats)
    stats[picked] = sample(stats[picked], backends[picked], fun, range, args)
    return nothing
end

pick(stats) = argmin(draw(law) for law in stats)

# signature of function call
signature(x) = typeof(x)
signature(a::AbstractArray) = eltype(a), axes(a)
signature(t::Union{Tuple, NamedTuple}) = map(signature, t)

function sample(law, backend, fun::Fun, range, args) where Fun
    compile_time = Base.cumulative_compile_time_ns()
    Base.cumulative_compile_timing(true)
    start = time_ns()
    ManagedLoops.offload(fun, backend, range, args...)
    elapsed = (time_ns()-start)*1e-9
    Base.cumulative_compile_timing(false)
    if Base.cumulative_compile_time_ns() == compile_time # no time spent compiling
        return push(law, elapsed)
    else
        return law
    end
end
