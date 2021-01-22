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
end

ermentrout_monotonic_params = HE2018Params(;
    Aee=1., Aie=1., Aei=1.5, Aii=0.25,
    θef=0.125, θif=0.4, τ=0.4, 
    aef=50., aif=50.,
    decaye=1., decayi=1.

)

ermentrout_depblock_params = HE2018DepParams(;
    Aee=1., Aie=1.2, Aei=1.5, Aii=0.25,
    θef=0.125, θif=0.4, θib=0.7, τ=0.4, 
    aef=50., aif=50., aib=50.,
    decaye=0.5, decayi=1.
)

tws_params = WCMDepParams(;
    Aee=70., Aie=35., Aei=70., Aii=70.,
    θef=6., θif=7., θib=1., τ=1., 
    aef=1., aif=1., aib=1.,
    decaye=1., decayi=1.
)

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

# FIXME
# function wcm_du_defn(u, v, p::Union{WCMDepParams,WCMParams})
#     @unpack Aee, Aei, θef, aef, decaye = p
#     du = Aee * u - Aei * v
#     du = (1-u) * simple_sigmoid_fn(du, aef, θef) - decaye * u
#     du
# end

# function wcm_dv_defn(u, v, p::WCMParams)
#     @unpack  Aie, Aii, τ, θif, aif, decayi = p
#     dv = Aie * u - Aii * v
#     dv = (1-v) * (simple_sigmoid_fn(dv, aif, θif)) - decayi * v
#     dv /= τ
#     dv
# end

# function wcm_dv_defn(u, v, p::WCMDepParams)
#     @unpack  Aie, Aii, τ, θif, θib, aif, aib, decayi = p
#     dv = Aie * u - Aii * v
#     dv = (1-v) * (simple_sigmoid_fn(dv, aif, θif) - simple_sigmoid_fn(dv, aib, θib)) - decayi * v
#     dv /= τ
#     dv
# end

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
