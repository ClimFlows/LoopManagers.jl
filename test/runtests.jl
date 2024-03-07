using LoopManagers
using ManagedLoops: @loops, @vec
using SIMDMathFunctions

using InteractiveUtils: versioninfo, @code_native
using Chairmarks: @b, @be
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
    display(@be loop!(mgr, exp, a, b))
    return nothing
end

managers = Any[
    nothing,
    LoopManagers.PlainCPU(),
    LoopManagers.VectorizedCPU(),
    LoopManagers.MultiThread(),
    LoopManagers.MultiThread(VectorizedCPU(4)),
]

@testset "LoopManagers.jl" begin
    # Write your tests here.
    let b = randn(1024, 1024)
        for mgr in managers
            test(mgr, b)
            @test true
        end
    end

end
