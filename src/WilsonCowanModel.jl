module WilsonCowanModel

using Parameters
using StaticArrays
using Simulation73
import Simulation73: target_loss
using Random
using MacroTools, IterTools
using TensorOperations
using BioNeuralNetworkModels

export WCMSpatial

export AbstractStimulus, SharpBumpStimulus, NoisyStimulus, GaussianNoiseStimulus, NoStimulus

include("helpers.jl")
include("stimulus.jl")
include("models.jl")

# using Optim
# export MatchExample, StretchExample, SpatioTemporalFnTarget, @optim_st_target
# include("target.jl")

end #module
