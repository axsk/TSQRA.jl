# experiment
# sensivity of eigenvalues on interaction strength

logspace(start, stop, length) = exp.(range(log(start), log(stop), length))

function exp_sens(;
    grid=ngrid(3),
    strengths=0:0.1:1,
    clip=30,
    verbosity=1)

    vc = combined_system(grid)
    vi = interaction_only(grid)
    global evs = []

    @show fl = flux(grid)

    for strength in strengths
        @show strength
        v = combinesystems(vc, 1:9, vi * strength, [1, 2, 3, 7, 8, 9])
        d, e = tensor_sqra(v; clip)
        v, = eigenfuns(d, e, n=2; verbosity)
        v *= fl
        @show v
        push!(evs, v)
    end
    evs
end

function confirm_lucas_eigenvalues(grid=defaultgrid)
    flx = flux(grid)

    D, E = tensor_sqra(system1(grid), clip=Inf)
    v1 = eigenfuns(D, E)[1] * flx

    D, E = tensor_sqra(system2(grid), clip=Inf)
    v2 = eigenfuns(D, E)[1] * flx

    real.(v1), real.(v2)
end

function combine_evs(v1, v2)
    evs = Float64[]
    for v1 in v1, v2 in v2
        push!(evs, v1 + v2)
    end
    return sort(evs, rev=true)
end

function luca_evs_over_ngrid(nrange=4:2:10)
    evs = map(nrange) do n
        v1, v2 = confirm_lucas_eigenvalues(ngrid(n, 1))
        combine_evs(v1, v2)[2:5]
    end
    scatter(nrange, reduce(hcat, evs)', legend=false, title="eigenvalues of combined system for different grids", xlabel="nbins", ylabel="λ") |> display
    evs
end