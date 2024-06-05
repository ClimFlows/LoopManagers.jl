using LoopManagers
using ManagedLoops: @loops, @vec
using SIMDMathFunctions
using KernelAbstractions

using InteractiveUtils: versioninfo, @code_native
using Chairmarks: @be
using Test

versioninfo()

myfun(x) = @vec if x>0 log(x) else exp(x) end

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
    display(@be loop!(mgr, myfun, a, b) seconds=1)
    return nothing
end

@testset "OpenMP-like manager" begin
    main = LoopManagers.MainThread()
    LoopManagers.parallel(main) do worker
        x = LoopManagers.share(worker) do
            randn()
        end
        println("Thread $(Threads.threadid()) has drawn $x.")
    end
    @test true
end

@testset "SIMD, multithread and auto-tuned managers" begin
    managers = Any[
    LoopManagers.PlainCPU(),
    LoopManagers.VectorizedCPU(),
    LoopManagers.MultiThread(),
    LoopManagers.MultiThread(VectorizedCPU(4)),
    LoopManagers.MultiThread(VectorizedCPU(8)),
    LoopManagers.MultiThread(VectorizedCPU(16)),]
    openMP=LoopManagers.MainThread(VectorizedCPU())
    let b = randn(1023, 1023)
        auto = LoopManagers.tune(managers)
        for mgr in vcat(managers, openMP, auto)
            test(mgr, b)
            @test true
        end
    end
end

test_bc(mgr, a,b,c) = @. mgr[a] = log(exp(b)*exp(c))

@testset "Managed broadcasting" begin
    managers = Any[
    LoopManagers.PlainCPU(),
    LoopManagers.VectorizedCPU(),
    LoopManagers.MultiThread(),
    LoopManagers.MultiThread(VectorizedCPU())]

    for dims in (10000, (100,100), (100, 10, 10), (10,10,10,10))
        a, b, c = (randn(Float32, dims) for i=1:3)
        for mgr in managers
            test_bc(mgr, a, b, c)
            @test true
        end
    end
end
