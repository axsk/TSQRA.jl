using LinearAlgebra
using Strided

const kB = 0.008314463
beta(T=300) = 1 / kB / T
DiffusionConstant(T=300, mass=1, gamma=1) = kB * T / mass / gamma
flux(dx, D=DiffusionConstant()) = D / dx^2
flux(dx::AbstractRange, D) = flux(step(dx), D)

function tensor_sqra(v::Array; beta=beta(), clip=Inf)
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
    apply_Qo(ones(length(D)), vec(D), size(D))
end

# VALUEFIX
Di(D) = replace(D, 0 => convert(eltype(D), Inf))

apply_Q(x, E, D) = reshape(apply_Q(vec(x), vec(E), vec(D), size(D)), size(x))
apply_AE(x, E) = reshape(apply_AE(vec(x), vec(E), size(D)), size(x))

function apply_Q(x::AbstractVector, E::AbstractVector, D::AbstractVector, s)
    y = apply_Qo(x, D, s)
    y .-= E .* x
    return y
end

function apply_Qo(x::AbstractVector, D::AbstractVector, s)
    y = vec(apply_A_banded(x .* D, s)) ./ Di(D)
end

function apply_AE(x::AbstractVector, E::AbstractVector, s)
    y = vec(apply_A_banded(x, s))
    y .-= E .* x
    return y
end

function modal_sum!(x, y, modes)
    s = ones(Int, length(size(x)))
    s[modes] .= size(y)
    y = reshape(y, s...)
    x .+= y
    return x
end

using SparseArrays
function reconstruct_matrix_sparse(action, len; maxcol=len)
    A = spzeros(len, len)
    for i in 1:maxcol
        x = zeros(len)
        x[i] = 1
        A[:, i] = action(x)
    end
    return A
end

function sparse_Q(D, E, maxcol=10)
    Q(x) = apply_Q(x, vec(E), vec(D), size(D))
    reconstruct_matrix_sparse(Q, length(D); maxcol)
end

