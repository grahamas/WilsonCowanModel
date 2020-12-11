using Contour, Parameters
using Makie, AbstractPlotting.MakieLayout
using NeuralModels: simple_sigmoid_fn

us = -0.1:0.01:1.0; vs = copy(us);

@with_kw struct WCMParams
    Aee
    Aei
    Aie
    Aii
    θef
    θif
    θib
    τ
    β
    decaye
    decayi
end

function lifted_wcm_param(;
    Aee=1., Aie=1., Aei=1.5, Aii=0.25,
    θef=0.125, θif=0.4, θib=7., τ=0.4, β=50.,
    decaye=1., decayi=1.)
    @lift WCMParams(;
            Aee=$(Node(Aee)), Aie=$(Node(Aie)), Aei=$(Node(Aei)), Aii=$(Node(Aii)),
            θef=$(Node(θef)), θif=$(Node(θif)), θib=$(Node(θib)), τ=$(Node(τ)), β=$(Node(β)),
            decaye=$(Node(decaye)), decayi=$(Node(decayi))
        )
end
ermentrout_monotonic_params = WCMParams(;
    Aee=1., Aie=1., Aei=1.5, Aii=0.25,
    θef=0.125, θif=0.4, θib=7., τ=0.4, β=50.,
    decaye=1., decayi=1.

)

ermentrout_depblock_params = WCMParams(;
    Aee=1., Aie=1.2, Aei=1.5, Aii=0.25,
    θef=0.125, θif=0.4, θib=0.7, τ=0.4, β=50.,
    decaye=0.5, decayi=1.
)

tws_params = WCMParams(;
    Aee=70., Aie=35., Aei=70., Aii=70.,
    θef=6., θif=7., θib=1., τ=1., β=1.,
    decaye=1., decayi=1.
)

function wcm_du_defn(u, v, p)
    @unpack Aee, Aei, θef, β, decaye = p
    du = Aee * u - Aei * v
    du = (1-u) * simple_sigmoid_fn(du, β, θef) - decaye * u
    du
end

function wcm_dv_defn(u, v, p)
    @unpack  Aie, Aii, τ, θif, θib, β, decayi = p
    dv = Aie * u - Aii * v
    dv = (1-v) * (simple_sigmoid_fn(dv, β, θif) - simple_sigmoid_fn(dv, β, θib)) - decayi * v
    dv /= τ
    dv
end

function wcm(u, p, t)
    return [wcm_du_defn(u[1], u[2], p, t), wcm_dv_defn(u[1], u[2], p, t)]
end

function wcm_nullclines(args...)
    scene, layout = layoutscene()
    layout[1,1] =  wcm_nullclines!(scene, args...)
    return (scene, layout)
end

function wcm_nullclines!(scene, us, vs, p)

    layout = GridLayout()
    layout[1,1] = ax = LAxis(scene)

    dus = [wcm_du_defn(u, v, p) for u in us, v in vs]
    dvs = [wcm_dv_defn(u, v, p) for u in us, v in vs]
    u_nullclines = Contour.lines(Contour.contour(us, vs, dus, 0.))
    v_nullclines = Contour.lines(Contour.contour(us, vs, dvs, 0.))

    for line in u_nullclines
        xs, ys = coordinates(line)
        Makie.lines!(ax, xs, ys, color=:blue)
    end

    for line in v_nullclines
        xs, ys = coordinates(line)
        Makie.lines!(ax, xs, ys, color=:red, linestyle=:dash, linesize=5)
    end
    layout[1,0] = LText(scene, "v", tellheight=false)
    layout[end+1,2] = LText(scene, "u", tellwidth=false)
    return layout
end