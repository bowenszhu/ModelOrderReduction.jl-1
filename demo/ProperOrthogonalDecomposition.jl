using Symbolics, LinearAlgebra

n = 100
@variables t
a = randn(n)
x = exp.(a * t)

tspan = (0.0, 10.0)
m = 10
snapshots = Symbolics.value.(mapreduce(tᵢ -> substitute.(x, (Dict(t => tᵢ),)),
                                       hcat,
                                       range(tspan[1], tspan[2], m)))
covariance = Symmetric(snapshots' * snapshots)
vals, vecs = eigen(covariance)
projection = (snapshots * (@. vecs * (1 / √vals)'))'
reduced_model = simplify.(projection * x)