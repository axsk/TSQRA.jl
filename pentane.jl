using PyCall

precision = Float64

# load and wrap the forces from luca's code
@pyinclude("forces.py")

vbond(x) = precision(py"Vbond($x, 0, 1, par_bonds)")
vangle(x) = precision(py"Vangle($x, 0, 1, 2, par_angles)")
vdihedral(x) = precision(py"Vdihedral($x, 0, 1, 2, 3, par_dihedrals)[0]")
vclj(x) = precision(py"Vcoulomb($x, 0, 1, [1, -1], par_coulomb) + Vlj($x, 0, 1, par_lj)")

function maketensor(func, bond)
    map(Iterators.product(bond...)) do coords
        coords = reshape(collect(precision, coords), 3, :)
        func(coords)
    end
end

maketensor(func::Function, bonds::Vector) = map(b -> maketensor(func, b), bonds)


defaultgrid = range(-2, 2, 5)
D = py"D"
beta = py"beta"

function vtensor_system1(grid=defaultgrid)
    g = grid

    coords = (
        g, g, g,  # 1 2 3
        g, g, 0,  # 4 5 x
        0, 0, 0,  # x x x
        0, 1, 0)  # x x x

    v = zeros(precision, repeat([length(grid)], 5)...)

    forces = [
        (vbond, [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]),
        (vbond, [4, 5, 6, 7, 8, 9], [4, 5]),
        (vangle, [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5]),
        (vangle, [4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 5]),
        (vdihedral, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5]),
    ]

    for (f, c, modes) in forces
        t = maketensor(f, coords[c])
        modal_sum!(v, t, modes)
    end

    replace!(v, NaN => Inf)

    return v
end

function vtensor_system2(grid=defaultgrid)
    g = grid

    coords = (
        1, 0, 0,  # x x x
        0, 0, 0,  # x x x
        0, g, 0,  # x 6 x
        g, g, g)  # 7 8 9

    v = zeros(precision, repeat([length(grid)], 4)...)

    forces = [
        (vbond, [4, 5, 6, 7, 8, 9], [6]),
        (vbond, [7, 8, 9, 10, 11, 12], [6, 7, 8, 9]),
        (vangle, [4, 5, 6, 7, 8, 9, 10, 11, 12], [6, 7, 8, 9]),
        (vdihedral, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [6, 7, 8, 9]),
    ]

    for (f, c, modes) in forces
        t = maketensor(f, coords[c])
        modal_sum!(v, t, modes .- 5)  # .- 5 since we wrote the modes above for the combined system
    end

    replace!(v, NaN => Inf)

    return v
end

function combined_system(grid=defaultgrid)
    v1 = vtensor_system1(grid)
    v2 = vtensor_system2(grid)

    v = reshape(v1, size(v1)..., 1, 1, 1, 1) .+ reshape(v2, 1, 1, 1, 1, 1, size(v2)...)
end

function interacting_system(grid=defaultgrid;
    v=combined_system(grid))
    g = grid

    coords = (g, g, g, g, g, g)

    forces = [
        (vclj, [1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9])
    ]

    for (f, c, modes) in forces
        t = maketensor(f, coords[c])
        modal_sum!(v, t, modes)
    end

    replace!(v, NaN => Inf)

    return v
end

function tensor_sqra(v; D, beta, delta)
    exp.((-beta / 2) .* v) .* (D / delta^2)
end

function pentane_tensor2(grid=defaultgrid)
    v = interacting_system(grid)

    beta = py"beta"
    D = py"D"
    delta = step(grid)

    tensor_sqra(v; beta, D, delta)
end

function pentane_tensor(grid=defaultgrid)
    @show step(grid)
    g = grid

    # grids for the reduced systems sys1/sys2
    #                   # corresponding degree of freedom in the 9 particle system
    sys1 = (g, g, g,    # 1 2 3
        g, g, 0,        # 4 5 x
        0, 0, 0,        # x x x
        0, 1, 0)        # x x x

    sys2 = (1, 0, 0,    # x x x
        0, 0, 0,        # x x x
        0, g, 0,        # x 6 x
        g, g, g)        # 7 8 9

    # specification which grids span the bonds
    bonds = [
        sys1[[1, 2, 3, 4, 5, 6]],
        sys1[[4, 5, 6, 7, 8, 9]],
        sys2[[4, 5, 6, 7, 8, 9]],
        sys2[[7, 8, 9, 10, 11, 12]]
    ]

    # modes along which the forces apply in the final 9 particle system
    bondinds = [
        [1, 2, 3, 4, 5],
        [4, 5],
        [6],
        [6, 7, 8, 9]
    ]

    angles = [
        sys1[[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        sys1[[4, 5, 6, 7, 8, 9, 10, 11, 12]],
        sys2[[4, 5, 6, 7, 8, 9, 10, 11, 12]]
    ]

    angleinds = [
        [1, 2, 3, 4, 5],
        [4, 5],
        [6, 7, 8, 9]
    ]

    dihedrals = [
        sys1[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        sys2[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    ]

    dihedralinds = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9]
    ]

    coulomblj = [(g, g, g, g, g, g)]
    coulombljinds = [[1, 2, 3, 7, 8, 9]]

    # compute the energies
    @time "bonds" vb = maketensor(vbond, bonds)
    @time "angles" va = maketensor(vangle, angles)
    @time "dihedrals" vd = maketensor(vdihedral, dihedrals)
    @time "lj+coulomb" vc = maketensor(vclj, coulomblj)

    #global lj = vc
    #vc = []
    #coulombljinds = []

    potentials = vcat(vb, va, vd, vc)
    modes = vcat(bondinds, angleinds, dihedralinds, coulombljinds)

    foreach(potentials) do p
        @show extrema(filter(!isnan, vec(p)))
        # VALUEFIX
        replace!(p, NaN => Inf)  # NaN usually comes from 0/0, i.e. unphysical singularities
    end

    @show modes

    # add the potentials along the modes
    x = zeros(precision, repeat([length(grid)], maximum(reduce(vcat, modes)))...)
    @time "assembling full potential" foreach((t, i) -> modal_sum!(x, t, i), potentials, modes)

    # VALUEFIX
    #x[x.>10] .= Inf

    kB = 0.008314463                 # kJ mol-1 K
    T = 300                         # K   
    mass = 1                           # amu mol-1   (same mass for each atom)
    gamma = 1                           # ps-1 
    D = kB * T / mass / gamma       # nm2 ps-1
    sigma = sqrt(2 * D)              # nm ps-1/2
    beta = 1 / kB / T                  # kJ-1 mol 

    # sqra
    @time "sqra" D = exp.((-beta / 2) .* x) .* (D / step(g)^2)
    return D
end

function sqra_pentane(;
    D=@time "  computed D" pentane_tensor())

    # VALUEFIX
    #replace!(D, 0 => minimum(D[D.>0] / 2))

    one = zero(D) .+ 1
    @time "  computed E" E = Qo(one, D)

    return D, E
end

"coulomb + lennard jones"
function vclj_julia(x)
    # assuming k_ele =1
    # q[1] = 1, q[2] = -1
    d = norm(x[:, 1] - x[:, 2])
    vc = -1 / d
    vjl = 3^12 / d^12 - 2 * 3^6 / d^6
    return vc + vjl
end