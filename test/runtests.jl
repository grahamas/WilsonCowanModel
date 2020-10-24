using Test, WilsonCowanModel, Simulation73, NeuralModels,
  OrdinaryDiffEq

include("src/test_sanity.jl")

const ABS_STOP=300.0
const X_PROP = 0.9
example = (
                  N_ARR=1,N_CDT=1,P=2; 
                  SNR_scale=80.0, stop_time=ABS_STOP,
                  Aee=70.0, See=25.0,
                  Aii=2.0, Sii=27.0,
                  Aie=35.0, Sie=25.0,
                  Aei=70.0, Sei=27.0,
                  n_lattice=512, x_lattice=1400.0, 
                  aE=1.2, θE=6.0,
                  aI=1.0, θI=11.4,
                  stim_strength=6.0,
                  stim_radius=14.0,
                  stim_duration=7.0,
                  pop_names = ("E", "I"),
                  velocity_threshold=1e-7,
                  n_traveling_frames_threshold=50,
                  α = (1.0, 1.0),
                  β = (1.0, 1.0),
                  τ = (3.0, 3.0),
                  nonlinearity = pops(RectifiedSigmoidNonlinearity;
                      θ = [θE, θI],
                      a = [aE, aI]
                  ),
                  stimulus = pops(CircleStimulusParameter;
                      strength = [stim_strength, stim_strength],
                      radius = [stim_radius, stim_radius],
                      time_windows = [[(0.0, stim_duration)], [(0.0, stim_duration)]],
                      baseline=[0.0, 0.0]
                  ),
                  connectivity = FFTParameter(pops(GaussianConnectivityParameter;
                      amplitude = [Aee -Aei;
                                   Aie -Aii],
                      spread = [(See,) (Sei,);
                                (Sie,) (Sii,)]
                     )
                  ),
                  space = PeriodicLattice{Float64,N_ARR}(; n_points=(n_lattice,), 
                                                           extent=(x_lattice,)),
                  tspan = (0.0,stop_time),
                  algorithm=Tsit5(),
                  save_idxs=nothing,
                  other_opts...
            ) -> begin
    Simulation(
        WCMSpatial{N_CDT,P}(;
            pop_names = pop_names,
            α = α,
            β = β,
            τ = τ,
            nonlinearity = nonlinearity,
            stimulus = stimulus,
            connectivity = connectivity
           );
        space = space,
        tspan = tspan,
        algorithm = algorithm,
        save_idxs = save_idxs,
        other_opts...
    )
end



@testset "Integration test" begin
	@test example() != 0
end
