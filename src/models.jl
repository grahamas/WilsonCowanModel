# NOTE: P is currently the TRAILING dimension. I haven't revised comments yet.

# Rename to remove N redundancy
struct WCMSpatial{T,N_CDT,P,
        SCALARS<:NTuple{N_CDT,T},
        CONN<:PopInteractParam{P,<:AbstractConnectivityParameter{T,N_CDT}},
        NONL<:PopAct{P,<:AbstractNonlinearity{T}},
        STIM<:PopActParam{P,<:AbstractStimulusParameter{T}}
    } <: AbstractModel{T,N_CDT,P}
    α::SCALARS
    β::SCALARS
    τ::SCALARS
    connectivity::CONN
    nonlinearity::NONL
    stimulus::STIM
    pop_names::NTuple{P,String}
    function WCMSpatial(α::SCALARS,β::SCALARS,τ::SCALARS,
                        conn::CONN,nonl::NONL,stim::STIM,
                        pop_names::NTuple{P,String}) where {
            T,N_CDT,P,
            SCALARS<:NTuple{N_CDT,T},
            CONN<:PopInteractParam{P,<:AbstractConnectivityParameter{T,N_CDT}},
            NONL<:PopAct{P,<:AbstractNonlinearity{T}},
            STIM<:PopActParam{P,<:AbstractStimulusParameter{T}}
        }
        new{T,N_CDT,P,SCALARS,CONN,NONL,STIM}(α,β,τ,conn,nonl,stim,pop_names)
    end
end
struct WCMSpatialAction{T,N_CDT,P,
        SCALARS<:NTuple{N_CDT,T},
        CONN<:PopInteract{P,<:AbstractConnectivityAction{T,N_CDT}},
        NONL<:PopAct{P},#,<:AbstractNonlinearity{T}},
        STIM<:PopAct{P}#,<:AbstractStimulusAction{T,N_CDT}}
        } <: AbstractSpaceAction{T,N_CDT}
    α::SCALARS
    β::SCALARS
    τ::SCALARS
    connectivity::CONN
    nonlinearity::NONL
    stimulus::STIM
    pop_names::NTuple{P,String}
    function WCMSpatialAction(α::S,β::S,τ::S,
                        conn::CONN,nonl::NONL,stim::STIM,
                        pop_names::NTuple{P,String}) where {
            T,N_CDT,P,
            S<:NTuple{N_CDT,T},
            CONN<:PopInteract{P,<:AbstractConnectivityAction{T,N_CDT}},
            NONL<:PopAct{P,<:AbstractNonlinearity{T}},
            STIM<:PopAct{P,<:AbstractStimulusAction{T,N_CDT}}
        }
        new{T,N_CDT,P,S,CONN,NONL,STIM}(α,β,τ,conn,nonl,stim,pop_names)
    end
end

function WCMSpatial(; pop_names::NTuple{N_pops,String}, α::NTuple{N_pops,T}, 
        β::NTuple{N_pops,T}, τ::NTuple{N_pops,T}, 
        connectivity::PopInteractParam{N_pops,C}, nonlinearity::PopAct{N_pops,NL}, stimulus::PopActParam{N_pops,S}
        ) where {T,N_pops,C<:AbstractConnectivityParameter{T},NL<:AbstractNonlinearity{T},S<:AbstractStimulusParameter{T}}
    PopParams = (x) -> PopulationParameters(x...)
    WCMSpatial(
        α, β, τ,
        connectivity, nonlinearity,
        stimulus, pop_names
    )
end
function (wcm::WCMSpatial)(space::AbstractSpace)
    return WCMSpatialAction(wcm.α, wcm.β, wcm.τ,
    wcm.connectivity(space), wcm.nonlinearity, wcm.stimulus(space), wcm.pop_names)
end

# TODO: parametric annotations
function (wcm::WCMSpatialAction{T,N,P})(dA,A,p,t) where {T,N,P}
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
        
