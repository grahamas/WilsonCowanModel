using Parameters
using NeuralModels: simple_sigmoid

abstract type AbstractNullclineParams{T} end
abstract type AbstractWCMNullclineParams{T} <: AbstractNullclineParams{T} end
abstract type AbstractWCMMonNullclineParams{T} <: AbstractWCMNullclineParams{T} end
abstract type AbstractWCMDepNullclineParams{T} <: AbstractWCMNullclineParams{T} end
abstract type AbstractWCMErfNullclineParams{T} <: AbstractWCMNullclineParams{T} end

export AbstractNullclineParams, AbstractWCMMonNullclineParams, AbstractWCMDepNullclineParams, AbstractWCMErfNullclineParams,
AbstractWCMNullclineParams
export WCMDepParams, WCMMonParams, HE2018Params, HE2018DepParams

us = -0.1:0.01:1.0; vs = copy(us);

@with_kw struct WCMDepParams{T} <: AbstractNullclineParams{T}
    Aee::T
    Aei::T
    Aie::T
    Aii::T
    θef::T
    aef::T
    θif::T
    aif::T
    θib::T
    aib::T
    τ::T
    decaye::T
    decayi::T
    nonl_norm::T
end

@with_kw struct WCMDepErfParams{T} <: AbstractWCMErfNullclineParams{T}
    Aee::T
    Aei::T
    Aie::T
    Aii::T
    μef::T
    σef::T
    μif::T
    σif::T
    μib::T
    σib::T
    τ::T
    decaye::T
    decayi::T
    # nonl_norm::T
end

@with_kw struct WCMMonParams{T} <: AbstractWCMMonNullclineParams{T}
    Aee::T
    Aei::T
    Aie::T
    Aii::T
    θef::T
    aef::T
    θif::T
    aif::T
    τ::T
    decaye::T
    decayi::T
end

@with_kw struct WCMMonErfParams{T} <: AbstractWCMErfNullclineParams{T}
    Aee::T
    Aei::T
    Aie::T
    Aii::T
    μef::T
    σef::T
    μif::T
    σif::T
    τ::T
    decaye::T
    decayi::T
end

@with_kw struct HE2018DepParams{T} <: AbstractWCMDepNullclineParams{T}
    Aee::T
    Aei::T
    Aie::T
    Aii::T
    θef::T
    aef::T
    θif::T
    aif::T
    θib::T
    aib::T
    τ::T
    decaye::T
    decayi::T
    nonl_norm::T
end

@with_kw struct HE2018Params{T} <: AbstractWCMMonNullclineParams{T}
    Aee::T
    Aei::T
    Aie::T
    Aii::T
    θef::T
    aef::T
    θif::T
    aif::T
    τ::T
    decaye::T
    decayi::T
end

function wcm_du_defn(u, v, p::Union{HE2018DepParams,HE2018Params})
    @unpack Aee, Aei, θef, aef, decaye = p
    du = Aee * u + Aei * v
    du = simple_sigmoid(du, aef, θef) - decaye * u
    du
end

function wcm_dv_defn(u, v, p::HE2018Params)
    @unpack  Aie, Aii, τ, θif, aif, decayi = p
    dv = Aie * u + Aii * v
    dv = simple_sigmoid(dv, aif, θif) - decayi * v
    dv /= τ
    dv
end

function wcm_dv_defn(u, v, p::HE2018DepParams)
    @unpack  Aie, Aii, τ, θif, θib, aif, aib, decayi, nonl_norm = p
    dv = Aie * u + Aii * v
    dv = nonl_norm * (simple_sigmoid(dv, aif, θif) - simple_sigmoid(dv, aib, θib)) - decayi * v
    dv /= τ
    dv
end

function wcm_du_defn(u, v, p::Union{WCMDepParams,WCMMonParams})
    @unpack Aee, Aei, θef, aef, decaye = p
    du = Aee * u + Aei * v
    du = (1-u) * simple_sigmoid(du, aef, θef) - decaye * u
    du
end

function wcm_dv_defn(u, v, p::WCMMonParams)
    @unpack  Aie, Aii, τ, θif, aif, decayi = p
    dv = Aie * u + Aii * v
    dv = (1-v) * (NeuralModels.rectified_unzeroed_sigmoid(dv, aif, θif)) - decayi * v
    dv /= τ
    dv
end

function wcm_dv_defn(u, v, p::WCMDepParams)
    @unpack  Aie, Aii, τ, θif, θib, aif, aib, decayi, nonl_norm = p
    dv = Aie * u + Aii * v
    dv = nonl_norm * (1-v) * (NeuralModels.rectified_unzeroed_sigmoid(dv, aif, θif) - NeuralModels.rectified_unzeroed_sigmoid(dv, aib, θib)) - decayi * v
    dv /= τ
    dv
end

function wcm_du_defn(u, v, p::Union{WCMDepErfParams,WCMMonErfParams})
    @unpack Aee, Aei, μef, σef, decaye = p
    du = Aee * u + Aei * v
    du = (1-u) * shifted_erf(du, σef, μef) - decaye * u
    du
end

function wcm_dv_defn(u, v, p::WCMMonErfParams)
    @unpack  Aie, Aii, τ, μif, σif, decayi = p
    dv = Aie * u + Aii * v
    dv = (1-v) * (shifted_erf(dv, σif, μif)) - decayi * v
    dv /= τ
    dv
end

function wcm_dv_defn(u, v, p::WCMDepErfParams)
    @unpack  Aie, Aii, τ, μif, μib, σif, σib, decayi = p
    dv = Aie * u + Aii * v
    # dv = nonl_norm * (1-v) * (shifted_erf(dv, σif, μif) - shifted_erf(dv, σib, μib)) - decayi * v
    dv = (1-v) * (shifted_erf(dv, σif, μif) - shifted_erf(dv, σib, μib)) - decayi * v
    dv /= τ
    dv
end

function wcm!(F, u, p, t)
    F[1] = wcm_du_defn(u[1], u[2], p)
    F[2] = wcm_dv_defn(u[1], u[2], p)
end

function wcm(u, p, t)
    x = [wcm_du_defn(u[1], u[2], p), wcm_dv_defn(u[1], u[2], p)]
    return x
end

function (t::Type{<:WCMDepErfParams})(wcm::AbstractWilsonCowanModel{T,1,2}) where T
    nullcline_params = Dict()
    nullcline_params[:Aee] = amplitude(wcm.connectivity.p11)
    # FIXME $20 says this is transposed
    nullcline_params[:Aei] = amplitude(wcm.connectivity.p12)
    nullcline_params[:Aie] = amplitude(wcm.connectivity.p21)
    nullcline_params[:Aii] = amplitude(wcm.connectivity.p22)
    nullcline_params[:μef] = wcm.nonlinearity.p1.μ
    nullcline_params[:σef] = wcm.nonlinearity.p1.σ
    firing_fn = get_firing_fn(wcm.nonlinearity.p2)
    blocking_fn = get_blocking_fn(wcm.nonlinearity.p2)
    #nullcline_params[:nonl_norm] = NeuralModels.calc_norm_factor(wcm.nonlinearity.p2)
    nullcline_params[:μif] = firing_fn.μ 
    nullcline_params[:μib] = blocking_fn.μ
    nullcline_params[:σif] = firing_fn.σ
    nullcline_params[:σib] = blocking_fn.σ
    nullcline_params[:τ] = wcm.τ[2] / wcm.τ[1]
    nullcline_params[:decaye] = wcm.α[1]
    nullcline_params[:decayi] = wcm.α[2]
    return t(; nullcline_params...)
end
function (t::Type{<:WCMMonErfParams})(wcm::AbstractWilsonCowanModel{T,1,2}) where T
    nullcline_params = Dict()
    nullcline_params[:Aee] = amplitude(wcm.connectivity.p11)
    # FIXME $20 says this is transposed
    nullcline_params[:Aei] = amplitude(wcm.connectivity.p12)
    nullcline_params[:Aie] = amplitude(wcm.connectivity.p21)
    nullcline_params[:Aii] = amplitude(wcm.connectivity.p22)
    nullcline_params[:μef] = wcm.nonlinearity.p1.μ
    nullcline_params[:σef] = wcm.nonlinearity.p1.σ
    nullcline_params[:μif] = wcm.nonlinearity.p2.μ
    nullcline_params[:σif] = wcm.nonlinearity.p2.σ
    nullcline_params[:τ] = wcm.τ[2] / wcm.τ[1]
    nullcline_params[:decaye] = wcm.α[1]
    nullcline_params[:decayi] = wcm.α[2]
    return t(; nullcline_params...)
end

function (t::Type{<:AbstractWCMDepNullclineParams})(wcm::AbstractWilsonCowanModel{T,1,2}) where T
    nullcline_params = Dict()
    nullcline_params[:Aee] = amplitude(wcm.connectivity.p11)
    # FIXME $20 says this is transposed
    nullcline_params[:Aei] = amplitude(wcm.connectivity.p12)
    nullcline_params[:Aie] = amplitude(wcm.connectivity.p21)
    nullcline_params[:Aii] = amplitude(wcm.connectivity.p22)
    nullcline_params[:θef] = wcm.nonlinearity.p1.θ
    nullcline_params[:aef] = wcm.nonlinearity.p1.a
    fsig = get_firing_fn(wcm.nonlinearity.p2)
    bsig = get_blocking_fn(wcm.nonlinearity.p2)
    nullcline_params[:nonl_norm] = NeuralModels.calc_norm_factor(wcm.nonlinearity.p2.dosp)
    nullcline_params[:θif] = fsig.θ 
    nullcline_params[:θib] = bsig.θ
    nullcline_params[:aif] = fsig.a
    nullcline_params[:aib] = bsig.a
    nullcline_params[:τ] = wcm.τ[2] / wcm.τ[1]
    nullcline_params[:decaye] = wcm.α[1]
    nullcline_params[:decayi] = wcm.α[2]
    return t(; nullcline_params...)
end
function (t::Type{<:AbstractWCMMonNullclineParams})(wcm::AbstractWilsonCowanModel{T,1,2}) where T
    nullcline_params = Dict()
    nullcline_params[:Aee] = amplitude(wcm.connectivity.p11)
    # FIXME $20 says this is transposed
    nullcline_params[:Aei] = amplitude(wcm.connectivity.p12)
    nullcline_params[:Aie] = amplitude(wcm.connectivity.p21)
    nullcline_params[:Aii] = amplitude(wcm.connectivity.p22)
    nullcline_params[:θef] = wcm.nonlinearity.p1.θ
    nullcline_params[:aef] = wcm.nonlinearity.p1.a
    nullcline_params[:θif] = wcm.nonlinearity.p2.θ 
    nullcline_params[:aif] = wcm.nonlinearity.p2.a
    nullcline_params[:τ] = wcm.τ[2] / wcm.τ[1]
    nullcline_params[:decaye] = wcm.α[1]
    nullcline_params[:decayi] = wcm.α[2]
    return t(; nullcline_params...)
end
(t::Type{<:AbstractNullclineParams})(sim::AbstractSimulation) = t(sim.model)

get_nullcline_params(sim::AbstractSimulation) = get_nullcline_params(sim.model)
function get_nullcline_params(model::WCMSpatial)
    if model.nonlinearity.p2 isa AbstractSigmoidNonlinearityParameter
        return WCMMonParams(model)
    elseif model.nonlinearity.p2 isa AbstractErfNonlinearityParameter
        return WCMMonErfParams(model)
    elseif model.nonlinearity.p2 isa AbstractDifferenceOfSigmoidsParameter
        return WCMDepParams(model)
    elseif model.nonlinearity.p2 isa AbstractDifferenceOfErfsParameter
        return WCMDepErfParams(model)
    else
        error("Unhandled nonlinearity $(typeof(model.nonlinearity))")
    end
end
function get_nullcline_params(model::HarrisErmentrout2018)
    if model.nonlinearity.p2 isa AbstractSigmoidNonlinearityParameter
        return HE2018Params(model)
    elseif model.nonlinearity.p2 isa AbstractDifferenceOfSigmoidsParameter
        return HE2018DepParams(model)
    else
        error("Unhandled nonlinearity $(typeof(model.nonlinearity))")
    end
end

export get_nullcline_params