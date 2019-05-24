# Rename to remove N redundancy
struct WCMSpatial{T,D,P,C<:AbstractConnectivity{T,D},
                            L<:AbstractNonlinearity{T},
                            S<:AbstractStimulus{T,D},
                            SP<:AbstractSpace{T,D}} <: AbstractModel{T,D,P}
    α::SVector{P,T}
    β::SVector{P,T}
    τ::SVector{P,T}
    space::SP
    connectivity::SMatrix{P,P,C}
    nonlinearity::SVector{P,L}
    stimulus::SVector{P,S}
    pop_names::SVector{P,String}
end

function WCMSpatial{T,N,P}(;
        pop_names::Array{Str,1}, α::Array{T,1}, β::Array{T,1},
        τ::Array{T,1}, space::SP, connectivity::Array{C,2}, nonlinearity::Array{L,1},
        stimulus::Array{S,1}
        ) where {
            T,P,N,Str<:AbstractString,C<:AbstractConnectivity{T},
            L<:AbstractNonlinearity{T},S<:AbstractStimulus{T},SP<:AbstractSpace{T,N}
        }
    WCMSpatial{T,N,P,C,L,S,SP}(SVector{P,T}(α), SVector{P,T}(β), SVector{P,T}(τ), space,
        SMatrix{P,P,C}(connectivity), SVector{P,L}(nonlinearity),
        SVector{P,S}(stimulus), SVector{P,Str}(pop_names)
    )
end

struct WCMPopulationData{T,N,A<:AbstractArray{T,N}} <: AbstractHomogeneousNeuralData{T,N}
    x::A
end
struct WCMPopulationsData{T,N,A<:AbstractArray{<:WCMPopulationData{T,N}}} <: AbstractHeterogeneousNeuralData{T,N}
    u::A
end
initial_value(wcm::WCMSpatial{T,D,P}) where {T,D,P} = WCMPopulationsData([zero(wcm.space) for i in 1:P])

function make_linear_mutator(model::WCMSpatial{T,N,P}) where {T,N,P}
    function linear_mutator!(dA::WCMPopulationsData{T,D}, A::WCMPopulationsData{T,D}, t::T) where D
        @views for i in 1:P
            dA[i] .*= model.β[i] .* (1.0 .- A[i])
            dA[i] .+= -model.α[i] .* A[i]
            dA[i] ./= model.τ[i]
        end
    end
end

function Simulation73.make_system_mutator(model::WCMSpatial)
    println("Making mutators...")
    stimulus_mutator! = make_mutator(model.stimulus, model.space)
    println("Made stimulus.")
    connectivity_mutator! = make_mutator(model.connectivity, model.space)
    println("Made connectivity.")
    nonlinearity_mutator! = make_mutator(model.nonlinearity)
    println("Made nonlinearity.")
    linear_mutator! = make_linear_mutator(model)
    println("Made linear.")
    function system_mutator!(dA, A, p, t)
        println("Stimulus mutating...")
        stimulus_mutator!(dA, A, t)
        println("done. Connectivity mutating...")
        connectivity_mutator!(dA, A, t)
        println("done. Nonlinearity mutating...")
        nonlinearity_mutator!(dA, A, t)
        println("done. Linear mutating...")
        linear_mutator!(dA, A, t)
        println("done with time $t")
    end
end

### Thoughts on connectivity
# TODO move implementation of connectivity op to connectivity.jl
# FFT is *much* faster, so ideally would use that. However, the inner
# dimensions pose a challenge. If the dimensions were entirely independent
# (consider the circle for dir tuning) we could probably just convolve the average
# on the circle, and add that to the square appropriately. However, we want the
# anisotropic connectivity also to decay with distance (just not as fast) so we
# can't simply take the average. Buttttt suppose Kd(x) is the kernel
# corresponding to the isotropic component of the anisotropic connectivity, i.e.
# long-space-constant exponential decay, ancd Kt(θ) is the kernel corresponding
# to the anisotropic direction tuning, i.e. short-space-constant (a.k.a
# short-angle-constant) exponential decay. In practice, then, while direction
# tuning may live on the circle, connectivity due to direction tuning lives on
# the hypercylinder, where the connectivity kernel is K(x,θ) = Kd(x) Kt(θ).

# Note: FFT of open surface (e.g. torus, circle) is just FFT of tiled.
