# tSqRA implementation from https://arxiv.org/abs/2311.09779 Appendix B.
# Tested on Julia 1.9

Vec = Vector
Tensor = Array
Grid = Union{Vector,AbstractRange}

""" apply_A(x::Tensor, dims::Tuple)
efficient computation of A*x exploiting the banded structure of A
where A is the adjacency matrix of a regular grid in `length(dims)` dimensions.
`dims` is the size of the grid in each dimension.
"""
function apply_A(x::Tensor, dims::Tuple)
    y = zeros(size(x))
    len = length(x)
    off = 1 # offset
    for cd in dims # current dimension length
        bs = off * cd # blocksize
        for i in 1:bs:len-off # blockstart
            to = i:i+bs-off-1
            y[to] .+= x[to.+off]
            y[to.+off] .+= x[to]
        end
        off = bs
    end
    return y::Tensor
end

""" compute_D(potentials, indices, grids)
given a list of potential functions, 
a list of indices specifying the modes to apply them to 
and a list of grids for each dimension,
generate the "diagonal tensor" D for the tSqRA as described in the paper
"""
function compute_D(
    potentials::Vec{Function}, # list of potentials
    indices::Vec{<:Vec}, # list of dimensions each potential acts on
    grids::Vec{<:Grid}, # list of grids for each dimension
    beta=1,)
    y = ones(length.(grids)...)
    for (v, inds) in zip(potentials, indices)
        p = map(Iterators.product(grids[inds]...)) do x
            exp(-1 / 2 * beta * v(x))
        end
        dims = ones(Int, length(grids))
        dims[inds] .= size(p) # specify broadcasting dimensions
        y .*= reshape(p, dims...) # elementwise mult. of potential
    end
    return y::Tensor
end

apply_Q(x, D, E) = apply_A(x .* D, size(E)) ./ D .- E .* x

""" Example with V(x,y) = x^2 + x^2*y^2 on the domain [-1,1]^2
For illustrative purposes we represent the potential
as sum of a 1 and 2 dimensional potential."""
function example()
    potentials = [x -> x[1] .^ 2, x -> x[1] .^ 2 .* x[2] .^ 2]
    indices = [[1], [1, 2]]

    grids = [range(-1, 1, 10), range(-1, 1, 10)]
    D = compute_D(potentials, indices, grids)
    E = apply_A(D, size(D)) ./ D
    x = rand(10, 10)
    Qx = apply_Q(x, D, E)
end