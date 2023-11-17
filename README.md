# TSQRA

Tensorized SqRA as described in [1].
Contains the application to an artifical 9-dimensional pentane (heavy atoms only after removing symmetries) on a regular grid of 10^9 cells.
The original force-fields and code by L. Donati were used as a reference and are contained in the `python` directory.

## Core functionality

`eigen.jl:`: Krylov eigensolver computing the TSQRA spectrum.
`apply_a.jl`: Different implementations for the application of the adjacency operator A
`tensorops.jl`: Application of Q, AE, .. and modal_sum!() for construction of the whole tensor from lower-order tensors
`pentane.jl`: Forcefields and computation of the two pentane subsystems, the whole system etc.
`experiments.jl`: Compute the sensivity analysis of the eigenvalues wrt. to interaction strength.

## See also
The supplementary material to [1], found at https://github.com/zib-cmd/article-tsqra contains a KISS implementation, as well as the python notebooks to create the article's data.

## References

[1] A. Sikorski, A. Niknejad, M. Weber, L. Donati. Tensor-SqRA: Modeling the Transition Rates of Interacting Molecular Systems in terms of Potential Energies. https://arxiv.org/abs/2311.09779. 2023
