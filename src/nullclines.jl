using Parameters
using NeuralModels: simple_sigmoid_fn

abstract type AbstractNullclineParams end
abstract type AbstractWCMParams <: AbstractNullclineParams end
abstract type AbstractWCMDepParams <: AbstractNullclineParams end

export AbstractNullclineParams, AbstractWCMParams, AbstractWCMDepParams
export WCMDepParams, WCMParams, HE2018Params, HE2018DepParams

us = -0.1:0.01:1.0; vs = copy(us);

@with_kw struct WCMDepParams <: AbstractWCMDepParams
    Aee
    Aei
    Aie
    Aii
    θef
    aef
    θif
    aif
    θib
    aib
    τ
    decaye
    decayi
    nonl_norm
end

@with_kw struct WCMParams <: AbstractWCMParams
    Aee
    Aei
    Aie
    Aii
    θef
    aef
    θif
    aif
    τ
    decaye
    decayi
end

@with_kw struct HE2018DepParams <: AbstractWCMDepParams
    Aee
    Aei
    Aie
    Aii
    θef
    aef
    θif
    aif
    θib
    aib
    τ
    decaye
    decayi
end

@with_kw struct HE2018Params <: AbstractWCMParams
    Aee
    Aei
    Aie
    Aii
    θef
    aef
    θif
    aif
    τ
    decaye
    decayi
    nonl_norm
end

function wcm_du_defn(u, v, p::Union{HE2018DepParams,HE2018Params})
    @unpack Aee, Aei, θef, aef, decaye = p
    du = Aee * u + Aei * v
    du = simple_sigmoid_fn(du, aef, θef) - decaye * u
    du
end

function wcm_dv_defn(u, v, p::HE2018Params)
    @unpack  Aie, Aii, τ, θif, aif, decayi = p
    dv = Aie * u + Aii * v
    dv = simple_sigmoid_fn(dv, aif, θif) - decayi * v
    dv /= τ
    dv
end

function wcm_dv_defn(u, v, p::HE2018DepParams)
    @unpack  Aie, Aii, τ, θif, θib, aif, aib, decayi = p
    dv = Aie * u + Aii * v
    dv = simple_sigmoid_fn(dv, aif, θif) - simple_sigmoid_fn(dv, aib, θib) - decayi * v
    dv /= τ
    dv
end

function wcm_du_defn(u, v, p::Union{WCMDepParams,WCMParams})
    @unpack Aee, Aei, θef, aef, decaye = p
    du = Aee * u + Aei * v
    du = (1-u) * simple_sigmoid_fn(du, aef, θef) - decaye * u
    du
end

function wcm_dv_defn(u, v, p::WCMParams)
    @unpack  Aie, Aii, τ, θif, aif, decayi = p
    dv = Aie * u + Aii * v
    dv = (1-v) * (NeuralModels.rectified_unzeroed_sigmoid_fn(dv, aif, θif)) - decayi * v
    dv /= τ
    dv
end

function wcm_dv_defn(u, v, p::WCMDepParams)
    @unpack  Aie, Aii, τ, θif, θib, aif, aib, decayi, nonl_norm = p
    dv = Aie * u + Aii * v
    dv = nonl_norm * (1-v) * (NeuralModels.rectified_unzeroed_sigmoid_fn(dv, aif, θif) - NeuralModels.rectified_unzeroed_sigmoid_fn(dv, aib, θib)) - decayi * v
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


function (t::Type{<:AbstractWCMDepParams})(wcm::WCMSpatial{T,1,2}) where T
    nullcline_params = Dict()
    nullcline_params[:Aee] = amplitude(wcm.connectivity.p11)
    # FIXME $20 says this is transposed
    nullcline_params[:Aei] = amplitude(wcm.connectivity.p12)
    nullcline_params[:Aie] = amplitude(wcm.connectivity.p21)
    nullcline_params[:Aii] = amplitude(wcm.connectivity.p22)
    nullcline_params[:θef] = wcm.nonlinearity.p1.θ
    nullcline_params[:aef] = wcm.nonlinearity.p1.a
    fsig = get_firing_sigmoid(wcm.nonlinearity.p2)
    bsig = get_blocking_sigmoid(wcm.nonlinearity.p2)
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
function (t::Type{<:AbstractWCMParams})(wcm::WCMSpatial{T,1,2}) where T
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

wcm_nullcline_params(sim::AbstractSimulation) = wcm_nullcline_params(sim.model)
function wcm_nullcline_params(model::WCMSpatial)
    if model.nonlinearity.p2 isa AbstractSigmoidNonlinearityParameter
        return WCMParams(model)
    elseif model.nonlinearity.p2 isa AbstractDifferenceOfSigmoidsParameter
        return WCMDepParams(model)
    else
        error("Unhandled nonlinearity $(typeof(model.nonlinearity))")
    end
end

he2018_nullcline_params(sim::AbstractSimulation) = he2018_nullcline_params(sim.model)
function he2018_nullcline_params(model::WCMSpatial)
    if model.nonlinearity.p2 isa AbstractSigmoidNonlinearityParameter
        return HE2018Params(model)
    elseif model.nonlinearity.p2 isa AbstractDifferenceOfSigmoidsParameter
        return HE2018DepParams(model)
    else
        error("Unhandled nonlinearity $(typeof(model.nonlinearity))")
    end
end

export wcm_nullcline_params, he2018_nullcline_params