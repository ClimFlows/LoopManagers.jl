using LoopManagers
using ManagedLoops: @loops, @vec
using Test

@loops function loop!(_, fun, a, b)
    let (irange, jrange) = axes(a)
        @inbounds @vec for i in irange, j in jrange
                a[i,j] = fun(b[i,j])
        end
    end
end

function test(mgr, b)
    a = similar(b)
    @info mgr
    loop!(mgr, exp, a, b)
    @time loop!(mgr, exp, a, b)
    return b
end

managers = Any[nothing, LoopManagers.PlainCPU(),
    LoopManagers.VectorizedCPU(),
    LoopManagers.MultiThread(),
    LoopManagers.MultiThread(VectorizedCPU(4)) ]

@testset "LoopManagers.jl" begin
    # Write your tests here.
    let b=randn(1000,1000)
        for mgr in managers
            test(mgr, b)
            @test true
        end
    end

end
