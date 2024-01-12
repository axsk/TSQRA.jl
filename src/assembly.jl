const kB = 0.008314463
getbeta(T=300) = 1 / kB / T
DiffusionConstant(T=300, mass=1, gamma=1) = kB * T / mass / gamma
flux(dx, D=DiffusionConstant()) = D / dx^2
flux(dx::AbstractRange, D) = flux(step(dx), D)

getpi(D) = (-2 / getbeta()) .* log.(D)

function tensor_sqra(v::Array; beta=getbeta(), clip=Inf)
    v = clippotential(v, clip)
    D = exp.((-beta / 2) .* v) #.* (D / delta^2)
    E = compute_E(D)
    return D, E
end

function clippotential(t, clip)
    t = replace(t, NaN => Inf)
    t[t.>clip] .= Inf
    return t
end

function compute_E(D)
    apply_Qo(ones(size(D)), D)
end

Vec = Vector
Tensor = Array
Grid = Union{Vector,AbstractRange}

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


""" modal_sum!(x,y,modes)

Widened hadamard sum.
Adds y to x along the dimensions specified in the array `modes`,
broadcasting over the other dimensions."""
function modal_sum!(x, y, modes)
    s = ones(Int, length(size(x)))
    s[modes] .= size(y)
    y = reshape(y, s...)
    x .+= y
    return x
end