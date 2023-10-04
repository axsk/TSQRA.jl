#TSQRA
tensorized sqra

Core functionality:
`eigen.jl:`: Krylov eigensolver computing the TSQRA spectrum.
`apply_a.jl`: Different implementations for the application of the adjacency operator A
`tensorops.jl`: Application of Q, AE, .. and modal_sum!() for construction of the whole tensor from lower-order tensors
`pentane.jl`: Forcefields and computation of the two pentane subsystems, the whole system etc.
`experiments.jl`: Compute the sensivity analysis of the eigenvalues wrt. to interaction strength