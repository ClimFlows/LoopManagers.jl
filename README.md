# LoopManagers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/LoopManagers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/LoopManagers.jl/dev/)
[![Build Status](https://github.com/ClimFlows/LoopManagers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/LoopManagers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/LoopManagers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/LoopManagers.jl)

LoopManagers is the companion package of [MangedLoops](https://github.com/ClimFlows/ManagedLoops.jl). It provides managers to execute loops with SIMD, on multiple threads or on GPUs. There is also a meta-manager that selects among a provided set of managers the one with the shortest execution time, on a per-function basis.

## Example

```
# Would belong to a 'provider' module, depending only on ManagedLoops

using ManagedLoops: @loops, @vec

@loops function loop!(_, a, b)
    let (irange, jrange) = axes(a)
        @vec for i in irange, j in jrange
            @inbounds a[i, j] = @fastmath exp(b[i, j])
        end
    end
end

# Belongs to a 'consumer' module/program, that requires LoopManagers to run

using LoopManagers: PlainCPU, VectorizedCPU, MultiThread
using SIMDMathFunctions # for vectorized exp
using BenchmarkTools
using InteractiveUtils

versioninfo() # check JULIA_EXCLUSIVE and JULIA_NUM_THREADS
scalar = PlainCPU()
simd = VectorizedCPU(8)
threads = MultiThread(simd)

b = randn(1024, 1024);
a = similar(b);

for mgr in (scalar, simd, threads)
    @info mgr
    display(@benchmark loop!($mgr, $a, $b))
end
```
