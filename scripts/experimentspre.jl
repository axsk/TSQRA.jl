
""" as part 1 in the article """
function example1(; nx=50, dt=1.0, nkoop=100, nstart=10, nchi=2, t0=0)
    V1(x) = (x^2 - 1)^2 + 1 * x
    V2(y) = 2 * y^2
    V12(x, y) = x * y
    Vc(x, c=1) = V1(x[1]) + V2(x[2]) + c * V12(x[1], x[2])

    grid = range(-3.4, 3.4, nx)

    potentials = [x -> V1(x[1]), x -> V2(x[1]), x -> V12(x[1], x[2])]
    indices = [[1], [2], [1, 2]]

    # TODO: adjust beta (and h?)
    beta = getbeta()
    h = step(grid)

    @time Q1 = SqraCore.sqra_grid(V1.(grid); beta, h) |> collect  # collect to get dense
    Q2 = SqraCore.sqra_grid(V2.(grid); beta, h) |> collect

    # compute isolated membership functions
    @time chi1 = PCCAPlus.pcca(Q1, nchi)[1]
    chi2 = PCCAPlus.pcca(Q2, nchi)[1]

    # compute combined membership function
    chi = stack(c1 .* c2' for c1 in eachcol(chi1) for c2 in eachcol(chi2))

    # compute coupled stationary density
    @time D = compute_D(potentials, indices, [grid, grid])

    # PRE for coupled rate matrix with combined memberships as intial guess
    @time Qc1 = pre(chi, D, dt, nstart, nkoop)

    # PRE 2
    Q = QTensor(D)
    chif = reshape(chi, :, size(chi)[end])
    @time Qc2 = pre2(Q, chif, dt, t0)

    # PCCA on full Q
    Qs = sparse_Q(D; beta)
    @time chic = pcca(Qs, size(chif, 2), solver=KrylovSolver())[1]
    @time Qc = pinv(chic) * Qs * chic

    NamedTuple(Base.@locals)
end



function example2(; nx=5, dt=1.0, nkoop=100, nstart=10, nchi=2, nsys=3, t0=0.0, beta=getbeta())
    V(x) = (x[1]^2 - 1)^2
    Vc(x) = abs2(x[1] - x[2]) / 2

    grid1 = range(-2, 2, nx)  # corresponds to lucas Nedges = 6, a = 2.5
    grid = fill(grid1, nsys)

    potentials = [[V for i in 1:nsys]; [Vc for i in 1:(nsys-1)]]
    indices = [[[i] for i in 1:nsys]; [[i, i + 1] for i in 1:(nsys-1)]]

    h = step(grid1)

    Q1 = SqraCore.sqra_grid(V.(grid1); beta, h) |> collect
    chi1 = PCCAPlus.pcca(Q1, nchi).chi

    allchis = alltensorprods((chi1 for i in 1:nsys)...)

    chi = allchis[:, [1]]
    chi = allchis

    # compute coupled stationary density
    D = compute_D(potentials, indices, grid, beta)
    Q = QTensor(D, beta)

    #Qc1 = pre(chi, D, dt, nstart, nkoop)
    Qc2 = pre2(Q, chi, dt, t0)

    if length(grid)^nsys < 1000
        (; Qs, chic, Qc) = Qc_full(Q, nchi^nsys)
    end

    #@exfiltrate
    NamedTuple(Base.@locals)
end

using Plots
function plot_dt_dependence!(; dts=[0.1, 1, 2, 5, 10, 15] .* 0.1, kwargs...)

    (; D, nstart, nkoop, Q, chi, t0) = NamedTuple(kwargs)

    p1(dt) = pre(chi, D, dt, nstart, nkoop)
    p2(dt) = pre2(Q, chi, dt, t0)

    @time x = stack(dts) do dt
        diag(p2(dt))
    end .|> real
    p2 = plot(dts, x', title="PRE2"; legend=false)
end

function Qc_full(Q, nc)
    Qs = sparse(Q)
    # could use direct eigensolve + pcca
    chic = pcca(Qs, nc, solver=KrylovSolver(), optimize=true)[1]
    Qc = pinv(chic) * Qs * chic
    return (; Qs, chic, Qc)
end

# compute the coupled chi directly from the coupled Q
function chicoup()
    Dcoup = compute_D(potentials[1:3], indices[1:3], grid, getbeta())
    Qcoup = sparse_Q(Dcoup) ./ getbeta()
    chicoup = pcca(Qcoup, 8, solver=KrylovSolver(), optimize=true)[1]
end

# reconstruction of lucas tau dependencies
function preluca(; Qs=Qs, Nd=nsys, chi=chicoup())
    KTAU = [0.1, 1, 2, 5, 10, 15] .* 0.1
    rates = zeros((2^Nd, 2^Nd, length(KTAU)))
    for (k, t) in enumerate(KTAU)
        tildeK = exp(collect(Qs) .* t)
        chi1 = tildeK * chi
        M = pinv(chi1) * chi
        hatQc = log(inv(M)) ./ t
        rates[:, :, k] = hatQc
    end

    plot()
    for i in 1:8
        plot!(KTAU, rates[i, i, :])
    end
    plot!() |> display
    return
end
