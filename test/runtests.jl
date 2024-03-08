using LoopManagers
using ManagedLoops: @loops, @vec
using SIMDMathFunctions

using InteractiveUtils: versioninfo, @code_native
using Chairmarks: @be
using Test

versioninfo()

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
    display(@be loop!(mgr, exp, a, b) seconds=1)
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
    let b = randn(1024, 1024)
        auto = LoopManagers.tune(managers)
        for mgr in vcat(managers, openMP, auto)
            test(mgr, b)
            @test true
        end
    end
end
