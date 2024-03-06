using LoopManagers
using ManagedLoops: @loops, @vec
using Test

@loops function loop!(_, fun, a, b)
    let (irange, jrange) = axes(a)
        @inbounds for j in jrange
            for i in irange
                a[i,j] = fun(b[i,j])
            end
        end
    end
end

function test(mgr, fun, b)
    a = similar(b)
    @info mgr
    loop!(mgr, fun, a, b)
    @time loop!(mgr, fun, a, b)
    return b
end

managers = Any[nothing, LoopManagers.PlainCPU(),
    LoopManagers.VectorizedCPU(),
    LoopManagers.MultiThread(),
    LoopManagers.MultiThread(VectorizedCPU()),
    LoopManagers.PlainCPU()]

@testset "LoopManagers.jl" begin
    # Write your tests here.
    let fun = exp, b=randn(100,100)
        for mgr in managers
            test(mgr, fun, b)
            @test true
        end
    end

end
