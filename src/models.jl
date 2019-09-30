# NOTE: P is currently the TRAILING dimension. I haven't revised comments yet.

# Rename to remove N redundancy
struct WCMSpatial{T,N_CDT,P,
        SCALARS<:AbstractPopulationActionsParameters{P,T},
        CONN<:AbstractPopulationInteractionsParameters{P,<:AbstractConnectivity{T,N_CDT}},
        NONL<:AbstractPopulationActionsParameters{P,<:AbstractNonlinearity{T}},
        STIM<:AbstractPopulationActionsParameters{P,<:AbstractStimulus{T,N_CDT}}
    } <: AbstractModel{T,N_CDT,P}
    α::SCALARS
    β::SCALARS
    τ::SCALARS
    connectivity::CONN
    nonlinearity::NONL
    stimulus::STIM
    pop_names::NTuple{P,String}
    function WCMSpatial(α::S,β::S,τ::S,
                        conn::CONN,nonl::NONL,stim::STIM,
                        pop_names::NTuple{P,String}) where {
            T,N_CDT,P,S<:AbstractPopulationParameters{P,T},
            CONN<:AbstractPopulationInteractionsParameters{P,<:AbstractConnectivityParameter{T,N_CDT}},
            NONL<:AbstractPopulationActions{P,<:AbstractNonlinearity{T}},
            STIM<:AbstractPopulationActionsParameters{P,<:AbstractStimulusParameter{T,N_CDT}}
        }
        new{T,N,P,S,CONN,NONL,STIM}(α,β,τ,conn,nonl,stim,pop_names)
    end
end
struct WCMSpatialAction{T,N_CDT,P,
        SCALARS<:NTuple{N,T},
        CONN<:AbstractPopulationInteractions{P,<:AbstractConnectivity{T,N_CDT}},
        NONL<:AbstractPopulationActions{P,<:AbstractNonlinearity{T}},
        STIM<:AbstractPopulationActions{P,<:AbstractStimulus{T,N_CDT}} <: AbstractAction{T,N_CDT}
    α::SCALARS
    β::SCALARS
    τ::SCALARS
    connectivity::CONN
    nonlinearity::NONL
    stimulus::STIM
    pop_names::NTuple{P,String}
    function WCMSpatial(α::S,β::S,τ::S,
                        conn::CONN,nonl::NONL,stim::STIM,
                        pop_names::NTuple{P,String}) where {
            T,N_CDT,P,
            S<:NTuple{N,T},
            CONN<:AbstractPopulationInteractions{P,<:AbstractConnectivity{T,N_CDT}},
            NONL<:AbstractPopulationActions{P,<:AbstractNonlinearity{T}},
            STIM<:AbstractPopulationActions{P,<:AbstractStimulus{T,N_CDT}} <: AbstractAction{T,N_CDT}
        }
        new{T,N,P,S,CONN,NONL,STIM}(α,β,τ,conn,nonl,stim,pop_names)
    end
end

function WCMSpatial(; pop_names::NTuple{N_pops,String}, α::NTuple{N_pops,T}, 
        β::NTuple{N_pops,T}, τ::NTuple{N_pops,T}, 
        connectivity::Array, nonlinearity::NTuple{N_pops}, stimulus::NTuple{N_pops}
        ) where {T,P}
    PopParams = (x) -> PopulationParameters(x...)
    WCMSpatial(
        α, β, τ,
        PopParams(connectivity), PopParams(nonlinearity),
        PopParams(stimulus), pop_names
    )
end
(wcm::WCMSpatial)(space::AbstractSpace) = WCMSpatialAction(wcm.α, wcm.β, wcm.τ,
    wcm.connectivity(space), wcm.nonlinearity, wcm.stimulus(space), pop_names)

function (wcm::WCMSpatialAction{T,N,P})(dA,A,t)
    wcm.stimulus(dA, A, t)
    wcm.connectivity(dA, A, t)
    wcm.nonlinearity(dA, A, t)
    for i in 1:P
        dAi = population(dA,i); Ai = population(A,i)
        dAi .*= wcm.β[i] .* (1.0 .- Ai)
        dAi .+= -wcm.α[i] .* Ai
        dAi ./= wcm.τ[i]
    end
end
        
