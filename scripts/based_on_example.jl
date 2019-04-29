using DrWatson
quickactivate(@__DIR__)

using WilsonCowanModel
using Simulation73
using Dates
using ArgParse

arg_settings = ArgParseSettings()
@add_arg_table arg_settings begin
    "--data-root", "-d"
        help = "Location in which to store output data"
        default = datadir()
    "--example-name"
        help = "Name of example defined in examples.jl"
    "--modifications-case"
        help = "Name of file specifying dict of modifications"
    "--plotspec-case"
        help = "Name of file specifying plots"
end
args = parse_args(arg_settings)
data_root = args[:data_root]
example_name = args[:example_name]
modifications_case = args[:modifications_case]
plotspec_case = args[:plotspec_case]

ENV["GKSwstype"] = "100" # For headless plotting (on server)
ENV["MPLBACKEND"]="Agg"
using Plots
pyplot()

if modifications_case != nothing
    modifications_path = joinpath(scriptdir(), "modifications", "$(modifications_case).jl")
    include(modifications_path) # defines modifications::Dict{Symbol,<:Any}
    @eval simulation = Examples.$(Symbol(example_name))(; modifications...)
else
    @eval simulation = Examples.$(Symbol(example_name))()
end

if plotspec_case != nothing
    plotspec_path = joinpath(scriptdir(), "plotspecs", "$(plotspec_case).jl")
    include(joinpath(scriptdir(), "plotspecs", plotspec_path)) # defines plotspecs
    plots_path = joinpath(plotsdir(), example_name))
    plots_prefix = "$(modifications_case)_$(plotspec_case)_$(Dates.now())_$(current_commit())"
    mkpath.([sim_output_path, plots_path])
else
    plots_prefix = ""
    plotspecs = []
end

sim_output_path = joinpath(data_root, "sim", example_name)

execution = execute(simulation)
execution_dict = @dict execution
@tagsave(joinpath(sim_output_path, "$(modifications_case)_$(Dates.now())_$(current_commit()).bson"), execution_dict, true)

plot_and_save.(plotspecs, Ref(execution), plots_path, plots_prefix)