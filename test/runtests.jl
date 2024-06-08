using LoopManagers
using ManagedLoops: @loops, @vec
using SIMDMathFunctions
using KernelAbstractions

using InteractiveUtils: versioninfo, @code_native
using Chairmarks: @be
using Test

versioninfo()

myfun(x) = @vec if x > 0
    log(x)
else
    exp(x)
end

@loops function loop!(_, fun, a, b)
    let (irange, jrange) = axes(a)
        @vec for i in irange, j in jrange
            @inbounds a[i, j] = fun(b[i, j])
        end
    end
end

@loops function my_cumsum!(_, a, b, ptop, fac)
    let (irange, jrange) = (axes(a, 1), axes(a, 2))
        nz = size(a,3)
        for j in jrange
            @vec for i in irange
                @inbounds a[i, j, nz] = ptop + (fac/2)*b[i, j, nz]
            end
        end
        for j in jrange
            for k = nz:-1:2
                @vec for i in irange
                    @inbounds a[i, j, k-1] = a[i, j, k] + (b[i, j, k-1] + b[i,j,k])*(fac/2)
                end
            end
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

timed(fun, N) = minimum(i -> (@timed fun()).time, 1:N)

println()
@info "====== Multi-thread scaling ======"

function scaling(fun, N)
    let b = randn(1023, 1023), a = similar(b)
        single = 1e9
        for nt = 1:Threads.nthreads()
            simd = LoopManagers.VectorizedCPU()
            mgr = LoopManagers.MultiThread(simd, nt)
            elapsed = timed(() -> fun(mgr), N)
            nt == 1 && (single = elapsed)
            @info "Efficiency with $nt Threads:\t $(single/nt/elapsed)"
        end
    end
end

@info "   simple loop"
let b = randn(1023, 1023), a = similar(b)
    scaling(100) do mgr
        loop!(mgr, myfun, a, b)
    end
end
println()

@info "   reverse cumsum"
let b = randn(128, 64, 30), a = similar(b)
    scaling(100) do mgr
        my_cumsum!(mgr, a, b, 1.0, 1.234)
    end
end
println()

exit() # FIXME

println()
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

println()
@testset "SIMD, multithread and auto-tuned managers" begin
    managers = Any[
        LoopManagers.PlainCPU(),
        LoopManagers.VectorizedCPU(),
        LoopManagers.MultiThread(),
        LoopManagers.MultiThread(VectorizedCPU(4)),
        LoopManagers.MultiThread(VectorizedCPU(8)),
        LoopManagers.MultiThread(VectorizedCPU(16)),
    ]
    openMP = LoopManagers.MainThread(VectorizedCPU())
    let b = randn(1023, 1023)
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
