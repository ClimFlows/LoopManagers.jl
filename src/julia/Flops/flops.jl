# borrowed from GFlops.jl by François Févotte  https://github.com/triscale-innov/GFlops.jl (MIT expat license)

module Flops
# import Statistics
# using Printf:           @printf
# using PrettyTables:     pretty_table

export @count_ops

include("overdub.jl")
include("counter.jl")
include("count_ops.jl")

end # module
