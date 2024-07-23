using LoopManagers
using ManagedLoops: @loops, @vec
using SIMDMathFunctions
using KernelAbstractions

using ThreadPinning
pinthreads(:cores)
threadinfo()

using InteractiveUtils: versioninfo, @code_native
using Chairmarks: @be
using Test

include("cumsum.jl")

versioninfo()

myfun(x) = @vec if x > 0
    exp(log(x))
else
    log(exp(x))
end

@loops function loop!(_, fun, a, b)
    let (irange, jrange) = axes(a)
        @vec for i in irange, j in jrange
            @inbounds a[i, j] = fun(b[i, j])
        end
    end
end

function test(mgr, b)
    a = similar(b)
    @info mgr
    loop!(mgr, exp, a, b)
    display(@be loop!(mgr, myfun, a, b) seconds = 1)
    return nothing
end

function timed(fun, N)
    fun()
    times = [(@timed fun()).time for _ = 1:N+10]
    sort!(times)
    return Float32(sum(times[1:N]) / N)
end

function scaling(fun, name, N, simd = VectorizedCPU())
    @info "====== Multi-thread scaling: $name ======"
    single = 1e9
    @info "Threads \t elapsed \t speedup \t efficiency"
    for nt = 1:Threads.nthreads()
        mgr = LoopManagers.MultiThread(simd, nt)
        elapsed = timed(() -> fun(mgr), N)
        nt == 1 && (single = elapsed)
        percent(x) = round(100x; digits = 0)
        speedup = single / elapsed
        @info "$nt \t\t $elapsed \t $(percent(speedup))% \t $(percent(speedup/nt))%"
    end
    println()
end

let b = randn(1023, 1023), a = similar(b)
    scaling("compute-bound loop", 100) do mgr
        loop!(mgr, myfun, a, b)
    end
end
println()

let b = randn(128, 64, 30), a = similar(b)
    for vlen in (4, 16, 64)
        scaling_cumsum(msg, fun!) =
            scaling("$msg vlen=$vlen", 1000, VectorizedCPU(vlen)) do mgr
                fun!(mgr, a, b, 1.0, 1.234)
            end
        scaling_cumsum("reverse cumsum_1", my_cumsum!)
        scaling_cumsum("reverse cumsum_2", my_cumsum2!)
        scaling_cumsum("reverse cumsum_3", my_cumsum3!)
    end
end

@testset "OpenMP-like manager" begin
    main = LoopManagers.MainThread(PlainCPU(), 10)
    @info "Testing MainThread with $(main.nthreads) threads."
    LoopManagers.parallel(main) do worker
        @info "Worker $(worker.id)"
        x = LoopManagers.share(randn, worker)
        println("Thread $(Threads.threadid()) has drawn $x.")
    end
    @test true
end

println()
@testset "SIMD, multithread and auto-tuned managers" begin
    managers = Any[
        LoopManagers.PlainCPU(),
        LoopManagers.VectorizedCPU(),
        LoopManagers.MultiThread(),
        LoopManagers.MultiThread(VectorizedCPU(8)),
        LoopManagers.MultiThread(VectorizedCPU(16)),
        LoopManagers.MultiThread(VectorizedCPU(32)),
    ]
    openMP = LoopManagers.MainThread(VectorizedCPU())
    let b = randn(Float32, 1023, 1023)
        auto = LoopManagers.tune(managers)
        for mgr in vcat(managers, openMP, auto)
            test(mgr, b)
            @test true
        end
    end
end

test_bc(mgr, a, b, c) = @. mgr[a] = log(exp(b) * exp(c))

println()
@testset "Managed broadcasting" begin
    managers = Any[
        LoopManagers.PlainCPU(),
        LoopManagers.VectorizedCPU(),
        LoopManagers.MultiThread(),
        LoopManagers.MultiThread(VectorizedCPU()),
    ]

    for dims in (10000, (100, 100), (100, 10, 10), (10, 10, 10, 10))
        a, b, c = (randn(Float32, dims) for i = 1:3)
        for mgr in managers
            test_bc(mgr, a, b, c)
            @test true
        end
    end
end
