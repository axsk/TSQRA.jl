### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ f6b554b4-cb26-4f7b-b54b-5d0136820969
begin 
	cd("/home/htc/bzfsikor/code/tsqra")
	Pkg.activate(".")
	using Pkg, Revise, Plots
	using KrylovKit, TSQRA
	include("scripts/experimentspre.jl")
end

# ╔═╡ 6bae4cf1-0755-4561-97ca-842b6b118586
c=0.5
# note if c=0 we have a degenerate spectrum, making the decomposition harder => leading to worse results


# ╔═╡ 155ac672-5747-429d-933a-4c3d304e8468


# ╔═╡ 72ac3f14-478f-4ec2-a321-9f19b54301a7
# ╠═╡ show_logs = false
begin
	ndim1 = 3
	s1 = exsystem(ngrid=5, nchi=2, ndim=ndim1   ;c)
	s2 = exsystem(ngrid=5, nchi=4, ndim=ndim1*2 ;c)
	Q = s2.Q
end;

# ╔═╡ 7c5a60f9-a0c0-46a2-952c-1cf17d4ee946
begin
	X1 = TSQRA.alltensorprods(s1.X, s1.X)
	X2 = s2.X
	XR = rand(size(X2)...)
end;

# ╔═╡ 209a6f52-3594-4a6b-a665-fa84660462c3
begin
	niters = [2,5,10,20, 30, 50,100,200,500]
	sumres = 1:2
	ylims = (1e-15,10)
	pargs = (; ylims=ylims, yaxis=:log, xaxis=:log, xlabel="niters", ylabel="err")
end

# ╔═╡ 7903c87a-44ef-47be-9ccc-b1338c86f175
begin
	T, vecs, vals, info = schursolve(Q, rand(size(Q,2)), 10, :LR, Arnoldi())
	scatter(real.(vals), title="eigenvalues of coupled sys")
end

# ╔═╡ 845bd271-4679-497a-8e41-bb713337c56f
function residuals(Q, x; nevecs=10, maxiter=100)
	T, vecs, vals, info = schursolve(Q, x, nevecs, :LR, Arnoldi(;maxiter))
	info.normres
end

# ╔═╡ f64169cf-01ab-46b8-b4ce-845ec2d12a17
begin
	maxiter=10
	nevecs = 4
	plotparms = (;yaxis=:log)#, ylims=(1e-15, 1e2))
end

# ╔═╡ be6f390e-491d-46ae-9dd0-afa0d1209b08
p1 = begin
	plot()
	foreach(eachcol(s2.X)[1:end]) do x0
		plot!(residuals(Q, x0; nevecs, maxiter))
	end
	plot!(;plotparms..., title="evecs", xlabel="residual number", ylabel="res. error")
end

# ╔═╡ 7a88c262-fc84-4836-98c5-030d42e9a883
p2 = begin
	plot()
	for x0 in cumsum(eachcol(s2.X))[1:end]
		plot!(residuals(Q, x0; nevecs, maxiter))
	end
	plot!(;plotparms..., title="cumsum", xlabel="residual number", ylabel="res. error")
end

# ╔═╡ 497d4fbc-5c6c-4413-80ab-b78aacbe7fdd
p3 = begin
	plot()
	for x0 in [rand(size(Q,1)) for i in 1:size(s2.X,2)]
		plot!(residuals(Q, x0; nevecs, maxiter))
	end
	plot!(;plotparms..., title="random x0", xlabel="residual number", ylabel="res. error")
end

# ╔═╡ f4f49dde-5d9c-4780-8445-ec3c1181a3f0
### Let us now check how the residual error (the sum over the first few eigenvecs) does improve with the number of kryloviters 

# ╔═╡ eef1e225-8307-43b7-aa44-d0fdf0b0e538
p4 = plot(niters,
	[map(niters) do maxiter 
		sum(residuals(Q, x0; maxiter)[sumres]) 
	end for x0 in cumsum(eachcol(X2)[2:end])],
	title = "cumsum prod"; pargs...)

# ╔═╡ 4646e895-0c35-4fa5-bd0a-f91a5e68886d
# starting with the eigenspace of the product system
p5 = plot(niters,
	[map(niters) do maxiter 
		sum(residuals(Q, x0; maxiter)[sumres]) 
	end for x0 in cumsum(eachcol(X1)[2:end])],
	title = "cumsum prod"; pargs...)

# ╔═╡ 93a5b2ce-c581-4972-9cb5-5400f06b594e
# starting randomly
p6=plot(niters,
	[map(niters) do maxiter 
		sum(residuals(Q, x0; maxiter)[sumres]) 
	end for x0 in eachcol(XR)],
	title = "random"; pargs...)

# ╔═╡ f27dd0cf-afdf-4462-9b99-f91810097ac8
plot(p4, p5 ,p6, legend=false)

# ╔═╡ Cell order:
# ╠═f6b554b4-cb26-4f7b-b54b-5d0136820969
# ╠═6bae4cf1-0755-4561-97ca-842b6b118586
# ╟─155ac672-5747-429d-933a-4c3d304e8468
# ╠═72ac3f14-478f-4ec2-a321-9f19b54301a7
# ╠═7c5a60f9-a0c0-46a2-952c-1cf17d4ee946
# ╠═209a6f52-3594-4a6b-a665-fa84660462c3
# ╠═f27dd0cf-afdf-4462-9b99-f91810097ac8
# ╠═7903c87a-44ef-47be-9ccc-b1338c86f175
# ╠═845bd271-4679-497a-8e41-bb713337c56f
# ╠═f64169cf-01ab-46b8-b4ce-845ec2d12a17
# ╠═be6f390e-491d-46ae-9dd0-afa0d1209b08
# ╠═7a88c262-fc84-4836-98c5-030d42e9a883
# ╠═497d4fbc-5c6c-4413-80ab-b78aacbe7fdd
# ╠═f4f49dde-5d9c-4780-8445-ec3c1181a3f0
# ╠═eef1e225-8307-43b7-aa44-d0fdf0b0e538
# ╠═4646e895-0c35-4fa5-bd0a-f91a5e68886d
# ╠═93a5b2ce-c581-4972-9cb5-5400f06b594e
