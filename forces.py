import numpy as np


kB     = 0.008314463                 # kJ mol-1 K
T      = 300                         # K   
mass   = 1                           # amu mol-1   (same mass for each atom)
gamma  = 1                           # ps-1 
D      = kB * T / mass / gamma       # nm2 ps-1
sigma  = np.sqrt(2 * D)              # nm ps-1/2
beta   = 1 / kB / T                  # kJ-1 mol 

def Vbond(X, i, j, par_bonds):
    rij      = X[:,i] - X[:,j]
    norm_rij = np.sqrt( np.sum( rij ** 2 ) )
    
    return 0.5 * par_bonds[0] * ( norm_rij - par_bonds[1] ) ** 2

def Vangle(X, i, j, k, par_angles):
    ri_x = X[0,i]
    ri_y = X[1,i]
    ri_z = X[2,i]

    ri   =  np.array([ri_x, ri_y, ri_z])

    rj_x = X[0,j]
    rj_y = X[1,j]
    rj_z = X[2,j]
    
    rj   =  np.array([rj_x, rj_y, rj_z])

    rk_x = X[0,k]
    rk_y = X[1,k]
    rk_z = X[2,k]

    rk   =  np.array([rk_x, rk_y, rk_z])

    rij  = ri - rj
    rkj  = rk - rj

    norm_rij = np.sqrt( np.sum( rij ** 2 ) )
    norm_rkj = np.sqrt( np.sum( rkj ** 2 ) )
    
    theta_ijk = np.arccos( np.dot(rij,rkj) / ( norm_rij * norm_rkj ) )
    
    return 0.5 * par_angles[0] * ( theta_ijk - par_angles[1] ) ** 2

def Vdihedral(X, i, j, k, l, par_dihedrals):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    p0 = X[:,i]
    p1 = X[:,j]
    p2 = X[:,k]
    p3 = X[:,l]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    psi = np.degrees(np.arctan2(y, x))
    
    return par_dihedrals[0] * np.cos( par_dihedrals[1] * psi + par_dihedrals[2] ), psi

def Vcoulomb(X, i, j, q, par_coulomb):
    rij      = X[:,i] - X[:,j]
    norm_rij = np.sqrt( np.sum( rij ** 2 ) )
    
    return par_coulomb ** 2 * q[i] * q[j] / norm_rij

def Vlj(X, i, j, par_lj):
    rij      = X[:,i] - X[:,j]
    norm_rij = np.sqrt( np.sum( rij ** 2 ) )

    return par_lj[0] * ( par_lj[1] / norm_rij ) ** 12 - 2 * par_lj[0] * ( par_lj[1] / norm_rij ) ** 6

    # Bond parameters
kb        = 1
r0        = 0.5
par_bonds = np.array([kb, r0])

# Angle parameters
ka         = 1
theta0     = np.pi / 3
par_angles = np.array([ka, theta0])

# Dihedral parameters
kd             = 1
periodicity    = 2
psi0           = np.pi / 3
par_dihedrals  = np.array([kd, periodicity, psi0])

# Coulomb parameters
kele        = 1
par_coulomb = kele
q           = np.array([+1, 0, 0, 0, -1])

# LJ parameters
eps        = 1
req        = 3
par_lj  = np.array([eps, req])

# Number of atoms
Na = 5

# Number of dimensions per atom
Nd = 3

# Total number of dimensions
N  = Na * Nd

# Number of bonds, angles, dihedrals
Nbonds     = 4
Nangles    = 3
Ndihedrals = 2

# bonds, angles, dihedrals
bonds      = np.array([[0,1], [1,2], [2,3], [3,4]])
angles     = np.array([[0,1,2], [1,2,3], [2,3,4]])
dihedrals  = np.array([[0,1,2,3], [1,2,3,4]])


# Coordinates
X = np.array([[[-2. , -1,  0,  1,  2],
               [ 0, -1,  0, -1,  0],
               [ 1,  0,  1,  0,  1]]])

def energy(X):
    # Energy
    E = 0

    for n in range(Nbonds):
        E += Vbond(X[0], bonds[n,0], bonds[n,1], par_bonds)

    for n in range(Nangles):
        E += Vangle(X[0], angles[n,0], angles[n,1], angles[n,2], par_angles)

    for n in range(Ndihedrals):
        E += Vdihedral(X[0], dihedrals[n,0], dihedrals[n,1], dihedrals[n,2], dihedrals[n,3], par_dihedrals)[0]

        #Vdihedral(X2[c], dihedrals[0,0], dihedrals[0,1], dihedrals[0,2], dihedrals[0,3], par_dihedrals)[0]
    E += Vcoulomb(X[0], 0, 4, q, par_coulomb)
    E += Vlj(X[0], 0, 4, par_lj)
    return E

print(energy(X))