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
    display(@be loop!(mgr, exp, a, b))
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

managers = Any[
    LoopManagers.PlainCPU(),
    LoopManagers.VectorizedCPU(),
    LoopManagers.MultiThread(),
    LoopManagers.MultiThread(VectorizedCPU()),
    LoopManagers.MainThread(VectorizedCPU()),
]


@testset "SIMD and fork-join managers" begin
    let b = randn(1024, 1024)

        tuner = LoopManagers.tune(managers)
        let a = similar(b)
            for _ in 1:100
                loop!(tuner, exp, a, b)
            end
        end
        @info first(tuner.calls).second

        for mgr in vcat(managers, tuner)
            test(mgr, b)
            @test true
        end
    end
end
