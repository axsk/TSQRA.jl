using LinearAlgebra
using Strided

apply_DQDi(x, E) = apply_A(x) .- E .* x
apply_Q(x, E, D) = apply_A(D .* x) ./ D .- E .* x

function Qo(x, potentials::Vector)
    for (u, modes) in potentials
        hadamard!(x, u, modes)
    end
    x = apply_A(x)
    for (u, modes) in potentials
        hadamard!(x, u, modes, inv=true)
    end
    return x
end

function Qo(x, d::Array)
    x .*= d
    x = @time "applying A" apply_A(x)
    x ./= d
end




function hadamard!(x, y, modes; inv=false)
    s = ones(Int, length(size(x)))
    s[modes] .= size(y)
    y = reshape(y, s...)
    if inv
        x .= x ./ y
    else
        x .= x .* y
    end
end

function modal_sum!(x, y, modes)
    s = ones(Int, length(size(x)))
    s[modes] .= size(y)
    y = reshape(y, s...)
    x .+= y
    return x
end


function pentane()
    dims = repeat([10], 9)
    u1 = (rand(10), [1])
    i1 = (rand(10, 10), [2, 3])
    i2 = (rand(10, 10), [3, 4])
    potentials = [u1, i1, i2]  # here we need sqrt(exp(-beta u))
    one = ones(dims...)
    @time E = Qo(one, potentials)
end



#=

function apply_A_inplace(B)
    C = zero(B)
    for (mode, dim) in enumerate(size(B))
        A = zeros(dim, dim)
        A[diagind(A, -1)] .= 1
        A[diagind(A, 1)] .= 1

        IB = Tuple(i for i in 1:ndims(B) if i != mode)
        IC = ((mode, IB...), Tuple(1:ndims(B)))


        tensorcontract!(C, IC, A, ((1,), (2,)), :N, B, ((mode,), IB), :N, true, true)

        C += tensorcontract(IC, A, IA, B, IB)
    end
    return C
end

function apply_A(B)
    C = zero(B)
    off = [1, cumprod(size(B))...]
    off = [off..., -off...]
    t = zeros(Int, length(size(B)))
    for i in CartesianIndices(B)
        @inbounds for (mode, dim) in enumerate(size(B))
            t[mode] = 1
            j = CartesianIndex(t...)
            if i[mode] + 1 <= dim
                C[i] = B[i+j]
            end
            if i[mode] - 1 >= 1
                C[i] = B[i-j]
            end
            t[mode] = 0
        end
    end
    return C
end
=#