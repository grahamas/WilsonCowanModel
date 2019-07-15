# Rename to remove N redundancy
struct WCMSpatial{T,N_ARR,N_CDT,P,C<:AbstractConnectivity{T,N_CDT},
                            L<:AbstractNonlinearity{T},
                            S<:AbstractStimulus{T,N_CDT},
                            SP<:AbstractSpace{T,N_ARR,N_CDT}} <: AbstractModel{T}
    α::SVector{P,T}
    β::SVector{P,T}
    τ::SVector{P,T}
    space::SP
    connectivity::SMatrix{P,P,C}
    nonlinearity::SVector{P,L}
    stimulus::SVector{P,S}
    pop_names::SVector{P,String}
end

function WCMSpatial{T,N_ARR,N_CDT,P}(;
        pop_names::Array{Str,1}, α::Array{T,1}, β::Array{T,1},
        τ::Array{T,1}, space::SP, connectivity::Array{C,2}, nonlinearity::Array{L,1},
        stimulus::Array{S,1}
        ) where {
            T,P,N_ARR,N_CDT,Str<:AbstractString,C<:AbstractConnectivity{T},
            L<:AbstractNonlinearity{T},S<:AbstractStimulus{T},SP<:AbstractSpace{T,N_ARR,N_CDT}
        }
    WCMSpatial{T,N_ARR,N_CDT,P,C,L,S,SP}(SVector{P,T}(α), SVector{P,T}(β), SVector{P,T}(τ), space,
        SMatrix{P,P,C}(connectivity), SVector{P,L}(nonlinearity),
        SVector{P,S}(stimulus), SVector{P,Str}(pop_names)
    )
end

# struct WCMPopulationData{T,N,A<:AbstractArray{T,N}}
#     x::A
# end
# const WCMPopulationsData{T,N_ARR,N_CDT,P} = ArrayPartition{T,<:NTuple{P,<:WCMPopulationData{T,N}}}
const WCMPopulationData{T,N} = AbstractArray{T,N}
const WCMPopulationsData{T,N} = AbstractArray{T,N}
#Base.zero(data::DATA) where {T,N,A,DATA <: WCMPopulationData{T,N,A}}  = DATA(zero(data.x))
#Simulation73.initial_value(wcm::WCMSpatial{T,N_ARR,N_CDT,P}) where {T,N_ARR,N_CDT,P} = ArrayPartition([WCMPopulationData(zero(wcm.space)) for i in 1:P]...)
Simulation73.initial_value(wcm::WCMSpatial{T,N_ARR,N_CDT,P}) where {T,N_ARR,N_CDT,P} = zeros(T, P, size(wcm.space)...)

@memoize function make_linear_mutator(model::WCMSpatial{T,N_ARR,N_CDT,P}) where {T,N_ARR,N_CDT,P}
    function linear_mutator!(dA::PopsData, A::PopsData, t::T) where {T,PopsData <: WCMPopulationsData}
        @views for i in 1:P
            dAi = population(dA,i); Ai = population(A,i)
            dAi .*= model.β[i] .* (1.0 .- Ai)
            dAi .+= -model.α[i] .* Ai
            dAi ./= model.τ[i]
        end
    end
end

@memoize function memoized_make_mutator(args...)
    make_mutator(args...)
end

function Simulation73.make_system_mutator(model::WCMSpatial)
    stimulus_mutator! = memoized_make_mutator(model.stimulus, model.space)
    connectivity_mutator! = memoized_make_mutator(model.connectivity, model.space)
    nonlinearity_mutator! = memoized_make_mutator(model.nonlinearity)
    linear_mutator! = make_linear_mutator(model)
    function system_mutator!(dA, A, p, t)
        dA .= zero(eltype(dA))
        stimulus_mutator!(dA, A, t)
        connectivity_mutator!(dA, A, t)
        nonlinearity_mutator!(dA, A, t)
        linear_mutator!(dA, A, t)
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
