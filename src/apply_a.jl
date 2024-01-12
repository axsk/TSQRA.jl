""" compute the application of A by looping through all A=1 entries and adding the corresponding x entries """
function apply_A_banded!(y::AbstractVector, x::AbstractVector, dims::NTuple{N,Int}) where {N}
    y .= 0
    len = length(x)
    off = 1 # offset
    for cd in dims  # current dimension length
        bs = off * cd  # blocksize
        @inbounds for i in 1:bs:len-off
            bso = bs - off
            to = i:i+bso-1
            @views y[to] .+= x[to.+off]
            @views y[to.+off] .+= x[to]
        end
        off = bs
    end
    return y
end

#=

using AxisAlgorithms
using TensorOperations: tensorcontract
using LinearAlgebra
using SparseArrays

# somehow manually reshaping saves time here
function apply_A_banded(x, s=Tuple(size(x)); kwargs...)
    x = vec(x)
    y = similar(x)
    apply_A_banded!(y, x, s)
    y = reshape(y, s)
end


apply_A(x) = apply_A_banded(x)
generate_local_A(dim) = generate_local_A_sparse(dim)

function apply_A_tensorcontract(B; generate_A=generate_local_A)
    C = zero(B)
    for (mode, dim) in enumerate(size(B))
        A = generate_A(dim)
        IA = [mode, 0]
        IB = collect(1:ndims(B))
        IC = collect(1:ndims(B))
        IC[mode] = 0
        C += tensorcontract(IC, A, IA, B, IB)
    end
    return C
end

function apply_A_axis(x; generate_A=generate_local_A)
    Ax = zero(x)
    tmp = zero(x)
    for (mode, dim) in enumerate(size(x))
        A = generate_A(dim)
        A_mul_B_perm!(tmp, A, x, mode)
        Ax .+= tmp
    end
    return Ax
end

function apply_A_axis_md(x; generate_A=generate_local_A)
    Ax = zero(x)
    tmp = zero(x)
    for (mode, dim) in enumerate(size(x))
        A = generate_A(dim)
        A = convert.(eltype(x), A)
        A_mul_B_md!(tmp, A, x, mode)
        Ax .+= tmp
    end
    return Ax
end

using FLoops
function apply_A_axis_threaded(x; generate_A=generate_local_A)
    @floop for (mode, dim) in collect(enumerate(size(x)))
        A = generate_A(dim)
        A = convert.(eltype(x), A)
        y = A_mul_B_md(A, x, mode)
        @reduce(Ax = zero(x) + y)
    end
    return Ax
end

function apply_A_axis_md_threaded(x; generate_A=generate_local_A)
    @floop for (mode, dim) in collect(enumerate(size(x)))
        A = generate_A(dim)
        A = convert.(eltype(x), A)
        y = A_mul_B_md(A, x, mode)
        @reduce(Ax = zero(x) + y)
    end
    return Ax
end



function generate_local_A_dense(dim)
    A = zeros(dim, dim)
    A[diagind(A, -1)] .= 1
    A[diagind(A, 1)] .= 1
    A
end

using SparseArrays
function generate_local_A_sparse(dim)
    A = spzeros(dim, dim)
    A[diagind(A, -1)] .= 1
    A[diagind(A, 1)] .= 1
    A
end




# too big for high dim
using SparseArrays
function grid_adjacency(dims::NTuple{N,Int} where {N})
    dims = reverse(dims) # somehow we have to take the krons backwards, dont know why
    k = []
    for i in 1:length(dims)
        x = [spdiagm(ones(Bool, s)) for s in dims] # identity in all dimensions
        x[i] = spdiagm(-1 => ones(Bool, dims[i] - 1), 1 => ones(Bool, dims[i] - 1)) # neighbour matrix in dimension i
        push!(k, kron(x...))
    end
    sum(k)
end




# try the same but with banded
bands_A(x...) = bands_A(Tuple(x))
function bands_A(dim::NTuple{N,Int}) where {N}
    bands = []
    offsets = []
    offset = 1
    for d in eachindex(dim)
        cd = dim[d]
        b = ones(prod(dim) - offset)
        bl = offset * cd
        for i in bl:bl:length(b)
            b[i-offset+1:i] .= 0
        end
        push!(bands, b)
        push!(offsets, offset)
        offset *= dim[d]
    end
    return bands, offsets
end


function benchmark_apply_a(x=rand(Float32, repeat([10], 9)...))
    @show Threads.nthreads()
    for A in [generate_local_A_dense, generate_local_A_sparse]
        println(A)
        for meth in [apply_A_tensorcontract, apply_A_axis, apply_A_axis_md,
            apply_A_axis_md_threaded, apply_A_axis_threaded]
            println(meth)
            #try
                @time meth(x, generate_A=A)
            #catch
            #end
        end
    end
end

function benchmark2(;
    dims=8,
    bins=8,
    x=rand(repeat([bins], dims)...)
)
    Q = grid_adjacency(size(x))
    display(@benchmark $Q * vec($x))
    display(@benchmark apply_A_axis($x))
    display(@benchmark apply_A_axis_md($x))
    display(@benchmark apply_A_banded($x))
    nothing

end

=#