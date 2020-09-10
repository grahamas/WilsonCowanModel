module WilsonCowanModel

using Parameters
using StaticArrays
using Simulation73
import Simulation73: target_loss
using NeuralModels

export WCMSpatial

export WCMPopulationData, WCMPopulationsData

include("helpers.jl")
include("models.jl")

# using Optim
# export MatchExample, StretchExample, SpatioTemporalFnTarget, @optim_st_target
# include("target.jl")

end #module
