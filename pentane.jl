using PyCall


# load and wrap the forces from luca's code
@pyinclude("forces.py")

vbond(x) = Float32(py"Vbond($x, 0, 1, par_bonds)")
vangle(x) = Float32(py"Vangle($x, 0, 1, 2, par_angles)")
vdihedral(x) = Float32(py"Vdihedral($x, 0, 1, 2, 3, par_dihedrals)[0]")

"coulomb + lennard jones"
function vclj(x)
    # assuming k_ele =1
    # q[1] = 1, q[2] = -1
    d = norm(x[:, 1] - x[:, 2])
    vc = -1 / d

end

# grid
ngrid = 11
grid = range(-1, 1, ngrid)
g = grid

# grids for the reduced systems sys1/sys2
#                   # corresponding degree of freedom in the 9 particle system
sys1 = (g, g, g,    # 1 2 3
    g, g, 0,        # 4 5 x
    0, 0, 0,        # x x x
    -2, 2, 0)        # x x x

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

function maketensor(func, bond)
    map(Iterators.product(bond...)) do coords
        coords = reshape(collect(Float32, coords), 3, :)
        func(coords)
    end
end

maketensor(func::Function, bonds::Vector) = map(b -> maketensor(func, b), bonds)

function pentane_tensor()
    # compute the energies
    @time "bonds" vb = maketensor(vbond, bonds)
    @time "angles" va = maketensor(vangle, angles)
    @time "dihedrals" vd = maketensor(vdihedral, dihedrals)

    potentials = vcat(vb, va, vd)
    foreach(potentials) do p
        replace!(p, NaN => Inf)
    end
    modes = vcat(bondinds, angleinds, dihedralinds)

    # add the potentials along the modes
    x = zeros(Float32, repeat([length(grid)], 9)...)
    @time "assembling full potential" foreach((t, i) -> modal_sum!(x, t, i), potentials, modes)

    # sqra
    @time "sqra" D = exp.(x ./ -2)
    return D
end


function sqra_pentane(
    D=@time "  computed D" pentane_tensor())

    #replace!(D, 0 => minimum(D[D.>0] / 2))

    one = zero(D) .+ 1
    @time "  computed E" E = Qo(one, D)

    return D, E
end
