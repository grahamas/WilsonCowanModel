using ModelingToolkit
using NeuralModels: simple_sigmoid_fn
using StaticArrays

struct SigmoidParams{T} <: FieldVector{2,T}
    θ::T
    a::T
end

struct DOSParams{T} <: FieldVector{4,T}
    θf::T
    θb::T
    af::T
    ab::T
end

struct ConnectivityParams{T} <: FieldVector{4,T}
    ee::T
    ei::T
    ie::T
    ii::T
end

struct EIParams{T} <: FieldVector{2,T}
    e::T
    i::T
end

@parameters t 
@parameters A[1:2,1:2], θf[1:2], a[1:2], θb
@parameters τ[1:2]
@parameters α[1:2]
@variables u(t) v(t)
D = Differential(t)

diff_of_sigmoids(x, af, θf, ab, θb) = simple_sigmoid_fn(x, af, θf) - simple_sigmoid_fn(x, ab, θb)
dos_norm_factor(θf, θb) = (θb - θf) / 2 + θf

wcm_depblock_eqns = [
    D(u) ~ ((1-u) * simple_sigmoid_fn(A[1,1] * u + A[1,2] * v, a[1], θf[1]) - α[1] * u) / τ[1],
    D(v) ~ ((1-v) * dos_norm_factor(θf[2], θb) * diff_of_sigmoids(A[2,1] * u + A[2,2] * v, a[2], θf[2], a[2], θb) - α[2] * v) / τ[2]
]

wcm_depblock_sys = ODESystem(wcm_depblock_eqns)

initial_state = [u => 0.5, v => 0.5]
ps = [
    α => [0.4, 0.7],
    θf => [0.12, 0.2], θb => 0.5,
    a => [50., 50.],
    A => [1.0 1.0;
         1.0 1.0],
    τ => [1.0, 0.4],
    t => 0.
]

odes = ODESystem(wcm_depblock_eqns, t, [u, v], [A, θf, a, θb, τ, α])
prob = ODEProblem(odes, initial_state, (0., 10.), ps)

using DifferentialEquations
sol = solve(prob, Tsit5())
# prob = NonlinearProblem(ns, guess, ps)