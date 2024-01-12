using LinearAlgebra
using Strided


# VALUEFIX
Di(D) = replace(D, 0 => convert(eltype(D), Inf))

apply_Q(x, D, E) = reshape(apply_Q!(similar(vec(x)), vec(x), vec(D), vec(E), size(D)), size(x))
apply_Qo(x, D) = reshape(apply_Qo!(similar(vec(x)), vec(x), vec(D), size(D)), size(x))
apply_AE(x, E) = reshape(apply_AE!(similar(vec(x)), vec(x), vec(E), size(E)), size(x))

function apply_Q!(y::V, x::V, D::V, E::V, s) where {V<:AbstractVector}
    y = apply_Qo!(y, x, D, s)
    y .-= E .* x
    return y
end

function apply_Qo!(y::V, x::V, D::V, s) where {V<:AbstractVector}
    y .= apply_A_banded!(y, x .* D, s)
    y ./= D  # TODO: should we use Di(D) here?
end

function apply_AE!(y::T, x::T, E::T, s) where {T}
    apply_A_banded!(y, x, s)
    y .-= E .* x
end



## Implementation of a tensor type dispathcing to banded operations

# this supports eg. ExponentialUtilities.expv
struct QTensor{T} #<: AbstractMatrix{eltype(T)}
    D::T
    E::T
    beta::eltype(T)
end

QTensor(D, beta=1) = QTensor(D, reshape(compute_E(D), size(D)), beta)

Base.eltype(::QTensor{T}) where {T} = eltype(T)
Base.size(Q::QTensor, dim) = length(Q.D)
function Base.size(Q::QTensor)
    n = length(Q.D)
    (n, n)
end
LinearAlgebra.ishermitian(Q::QTensor) = false

function LinearAlgebra.mul!(y, Q::QTensor, x::AbstractVector)
    apply_Q!(y, x, vec(Q.D), vec(Q.E), size(Q.E))
    return y ./= Q.beta
end

function LinearAlgebra.mul!(y, Q::QTensor, x::AbstractMatrix)
    for i in 1:size(x, 2)
        @views mul!(y[:, i], Q, x[:, i])
    end
    y
end

import Base.*
*(Q::QTensor, x) = mul!(similar(x), Q, x)

using SparseArrays
SparseArrays.sparse(Q::QTensor) = SqraCore.sqra_grid(getpi(Q.D), beta=Q.beta)


## The following are slow, meant for debugging only

function reconstruct_matrix_sparse(action, len)
    A = spzeros(len, len)
    for i in 1:len
        x = zeros(len)
        x[i] = 1
        A[:, i] = action(x)
    end
    return A
end

function sparse_Q(D; E=compute_E(D), beta=1)
    Q(x) = apply_Q(x, vec(D), vec(E), size(D))
    reconstruct_matrix_sparse(Q, length(D)) ./ beta
end