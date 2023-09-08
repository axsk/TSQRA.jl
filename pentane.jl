using PyCall

precision = Float64



# apply func to the coordinates in grid
function maketensor(func, grid::Tuple)
    map(Iterators.product(grid...)) do coords
        coords = reshape(collect(precision, coords), 3, :)
        func(coords)
    end
end

function system1(grid=defaultgrid)
    g = grid

    coords = (
        g, g, g,  # 1 2 3
        g, g, 0,  # 4 5 x
        0, 0, 0,  # x x x
        0, pi, 0)  # x x x

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
        pi, pi, 0,  # x x x
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

function interaction_only(grid=defaultgrid)
    g = grid
    t = maketensor(vclj, (g, g, g, g, g, g))
    replace!(t, NaN => Inf)
    t
end

function combined_system(grid=defaultgrid)
    v1 = system1(grid)
    v2 = system2(grid)

    v = reshape(v1, size(v1)..., 1, 1, 1, 1) .+ reshape(v2, 1, 1, 1, 1, 1, size(v2)...)
end


function interacting_system(grid=defaultgrid;
    v=combined_system(grid),
    clip=30,
    interaction=1)

    g = grid

    i = interaction_only(grid)
    n = size(i, 1)
    i = reshape(i, (n, n, n, 1, 1, 1, n, n, n))

    v .+= i * interaction

    return v
end




vbond(x) = vbondjl(x)
vangle(x) = vanglejl(x)
vdihedral(x) = vdihedraljl(x)
vclj(x) = vclj_julia(x)

function vbondjl(x; kb=1, r0=0.5)
    r = norm(x[:, 1] - x[:, 2])
    return 0.5 * kb * (r - r0)^2
end

function vanglejl(x; ka=1, theta0=2 / 3 * pi)
    a = x[:, 1] - x[:, 2]
    b = x[:, 3] - x[:, 2]
    theta = acos(clamp(a ⋅ b / (norm(a) * norm(b)), -1, 1))
    return 0.5 * ka * (theta - theta0)^2
end

function vdihedraljl(x; kd=1, periodicity=2, psi0=0)
    r1, r2, r3 = @views x[:, 2] - x[:, 1], x[:, 3] - x[:, 2], x[:, 4] - x[:, 3]
    b = r2
    u = cross(b, r1)
    w = cross(b, r3)
    psi = atan(cross(u, w)' * b, u' * w * norm(b))
    kd * cos(periodicity * psi + psi0)
end

"coulomb + lennard jones"
function vclj_julia(x; kele=1, charge=1, eps=1, req=1)
    # assuming k_ele =1
    # q[1] = 1, q[2] = -1
    d = @views norm(x[:, 1] - x[:, 2])
    coul = kele^2 * charge / d
    lenj = eps * ((req / d)^12 - 2 * (req / d)^6)
    return coul + lenj
end

testpython = false
if testpython

    # load and wrap the forces from luca's code
    @pyinclude("forces.py")

    vbondpy(x) = precision(py"Vbond($x, 0, 1, par_bonds)")
    vanglepy(x) = precision(py"Vangle($x, 0, 1, 2, par_angles)")
    vdihedralpy(x) = precision(py"Vdihedral($x, 0, 1, 2, 3, par_dihedrals)[0]")
    vcljpy(x) = precision(py"Vcoulomb($x, 0, 1, [1,1], par_coulomb) + Vlj($x, 0, 1, par_lj)")

    x = rand(3, 4)
    @assert vbondjl(x) ≈ vbondpy(x)
    @assert vanglejl(x) ≈ vanglepy(x)
    @assert vdihedraljl(x) ≈ vdihedralpy(x)
    @assert vclj_julia(x) ≈ vcljpy(x)
end