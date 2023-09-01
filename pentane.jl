using PyCall

precision = Float64

# load and wrap the forces from luca's code
@pyinclude("forces.py")

vbond(x) = precision(py"Vbond($x, 0, 1, par_bonds)")
vangle(x) = precision(py"Vangle($x, 0, 1, 2, par_angles)")
vdihedral(x) = precision(py"Vdihedral($x, 0, 1, 2, 3, par_dihedrals)[0]")
vclj(x) = precision(py"Vcoulomb($x, 0, 1, [1, -1], par_coulomb) + Vlj($x, 0, 1, par_lj)")

# apply func to the coordinates in grid
function maketensor(func, grid::Tuple)
    map(Iterators.product(grid...)) do coords
        coords = reshape(collect(precision, coords), 3, :)
        func(coords)
    end
end

defaultgrid = range(-2, 2, 5)
biggrid = range(-1.35, 1.35, 10)
D = py"D"
beta = py"beta"

function system1(grid=defaultgrid)
    g = grid

    coords = (
        g, g, g,  # 1 2 3
        g, g, 0,  # 4 5 x
        0, 0, 0,  # x x x
        0, 1, 0)  # x x x

    forces = [
        (vbond, [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]),
        (vbond, [4, 5, 6, 7, 8, 9], [4, 5]),
        (vangle, [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5]),
        (vangle, [4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 5]),
        (vdihedral, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5]),
    ]

    v = zeros(precision, repeat([length(grid)], 5)...)
    for (f, c, modes) in forces
        t = maketensor(f, coords[c])
        modal_sum!(v, t, modes)
    end

    replace!(v, NaN => Inf)
    return v
end

function system2(grid=defaultgrid)
    g = grid

    coords = (
        0.77, 0, 0,  # x x x
        0, 0, 0,  # x x x
        0, g, 0,  # x 6 x
        g, g, g)  # 7 8 9

    forces = [
        (vbond, 4:9, [6]),
        (vbond, 7:12, [6, 7, 8, 9]),
        (vangle, 4:12, [6, 7, 8, 9]),
        (vdihedral, 1:12, [6, 7, 8, 9]),
    ]

    v = zeros(precision, repeat([length(grid)], 4)...)
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

function interaction_only(grid=defaultgrid)
    g = grid
    t = maketensor(vclj, (g, g, g, g, g, g))
end

function interacting_system(grid=defaultgrid;
    v=combined_system(grid),
    clip=30)
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
    v[v.>clip] .= Inf

    return v
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