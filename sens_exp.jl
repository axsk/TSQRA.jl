# experiment
# sensivity of eigenvalues on interaction strength

function exp_sens(;
    grid=ngrid(3),
    strengths=0:0.1:1,
    clip=30,
    verbosity=2)

    vc = combined_system(grid)
    vi = interaction_only(grid)
    evs = []

    @show fl = flux(grid)

    for strength in strengths
        @show strength
        v = combinesystems(vc, 1:9, vi * strength, [1, 2, 3, 7, 8, 9])
        d, e = tensor_sqra(v; clip)
        v, = eigenfuns(d, e, n=2; verbosity)
        v *= fl
        @show v
        push!(evs, v[2])
    end
    strengths, evs
end

function combinesystems(sys1, bond1, sys2, bond2)
    totalsize = maximum(vcat(bond1, bond2))
    s1 = ones(Int, totalsize)
    s2 = ones(Int, totalsize)
    for (i, b) in enumerate(bond1)
        s1[b] = size(sys1, i)
    end
    for (i, b) in enumerate(bond2)
        s2[b] = size(sys2, i)
    end
    reshape(sys1, Tuple(s1)) .+ reshape(sys2, Tuple(s2))
end