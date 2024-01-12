module TSQRA

include("apply_a.jl")
include("tensorops.jl")
include("assembly.jl")

include("eigen.jl")

using LinearAlgebra: CartesianIndex, pinv
using Random: randexp

using SqraCore
using PCCAPlus
include("pre.jl")

#include("pentane.jl")
#include("experiments.jl")

end