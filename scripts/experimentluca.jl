using Dates: now

ngrid(n, a=1) = range(-a, a, n)
defaultgrid = ngrid(10, 1)

function run(;
    grid=defaultgrid,
    maxiter=30,
    v=@time "D" interacting_system(grid))

    @time "E" D, E = tensor_sqra(v)
    @time "λ" f = eigenfuns(D, E; maxiter)
    return D, E, f
end

lucaflux = 27.714876666666658

""" construct an initial guess for the interacting systems krylovspace
by computing the eigenfunction of the combined system """
function initialguess(grid=defaultgrid)
    efs = map([system1(grid), system2(grid)]) do sys
        d, e = tensor_sqra(sys)
        vals, efs, = eigenfuns(d, e, n=2)
        real.(efs[:, 2])
    end
    kron(efs...)
end

function visualize(x, g; kwargs...)
    c = [
        g[x[1]], g[x[2]], g[x[3]],  # 1 2 3
        g[x[4]], g[x[5]], 0,  # 4 5 x
        0, 0, 0,  # x x x
        0, g[x[6]], 0,  # x 6 x
        g[x[7]], g[x[8]], g[x[9]]]  # 7 8 9
    c = reshape(c, 3, :)
    plot!(eachrow(c)...; kwargs...)
    scatter!(eachrow(c)...; kwargs...)
end

function vis_combined_max(g=defaultgrid)
    ##D, _ = tensor_sqra(combined_system(g), clip=30)
    v1 = system1(g)
    v2 = system2(g)


    @show mx = vcat(Tuple(findmin(v1)[2])..., (Tuple(findmin(v2)[2]))...)
    visualize(mx, g)
end

function findnmin(v, n)
    v = copy(v)
    inds = []
    for i in 1:n
        @show m = findmin(v)
        ind = m[2]
        @show v[ind] = Inf
        push!(inds, ind)
    end
    return inds
end