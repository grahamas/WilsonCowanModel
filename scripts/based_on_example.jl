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
    "--no-save-raw"
        help = "Don't save raw simulation"
        action = :store_true
end
args = parse_args(ARGS, arg_settings)
@show args
data_root = args["data-root"]
example_name = args["example-name"]
modifications_case = args["modifications-case"]
plotspec_case = args["plotspec-case"]

ENV["GKSwstype"] = "100" # For headless plotting (on server)
ENV["MPLBACKEND"]="Agg"
using Plots
pyplot()

if modifications_case != nothing
    modifications_path = joinpath(scriptdir(), "modifications", "$(modifications_case).jl")
    include(modifications_path) # defines modifications::Dict{Symbol,<:Any}
    if (modifications isa Dict)
        modifications = [modifications]
    end
else
    modifications = [Dict()]
end

if plotspec_case != nothing
    plotspec_path = joinpath(scriptdir(), "plotspecs", "$(plotspec_case).jl")
    include(joinpath(scriptdir(), "plotspecs", plotspec_path)) # defines plotspecs
    plots_path = joinpath(plotsdir(), example_name)
    plots_path = joinpath(plots_path, "$(modifications_case)_$(plotspec_case)_$(Dates.now())_$(current_commit())")
    mkpath(plots_path)
else
    plotspecs = []
end
if !args["no-save-raw"]
    sim_output_path = joinpath(data_root, "sim", example_name, "$(modifications_case)_$(Dates.now())_$(current_commit())")
    mkpath(sim_output_path)
end

for modification in modifications
    simulation = Examples.get_example(example_name)(; modification...)
    execution = execute(simulation)
    mod_name = savename(modification; allowedtypes=(Real,String,Symbol,AbstractArray))
    if !args["no-save-raw"]
        execution_dict = @dict execution
        @tagsave("$(joinpath(sim_output_path, mod_name)).bson", execution_dict, true)
    end
    plot_and_save.(plotspecs, Ref(execution), plots_path, mod_name)
end
