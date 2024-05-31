#------------------------------
# setup
#------------------------------

cd(@__DIR__)

using Pkg;
Pkg.activate("../.")
Pkg.status()

using CSV
using Dates
using DataFrames
using Distributions
using Random
using GLM
using Flux
using LaTeXStrings
using LinearAlgebra
using Measures
using Parameters
using Plots 
using ProgressMeter
using StatsBase

gr()

sourcedir = "../src/"
include(joinpath(sourcedir, "load_data.jl"))
include(joinpath(sourcedir, "model.jl"))
include(joinpath(sourcedir, "ODE_solutions.jl"))
include(joinpath(sourcedir, "training.jl"))
include(joinpath(sourcedir, "eval_penalties.jl"))
include(joinpath(sourcedir, "plot_latent.jl"))

#------------------------------
# load data
#------------------------------

data_path = joinpath("../dataset/")

baseline_df = CSV.File(string(data_path, "baseline_df.csv"), truestrings = ["TRUE", "M"], falsestrings = ["FALSE", "F"], missingstring = ["NA"], decimal=',') |> DataFrame
timedepend_df = CSV.File(string(data_path, "timedepend_df.csv"), truestrings = ["TRUE"], falsestrings = ["FALSE"], missingstring = ["NA"], decimal=',') |> DataFrame

# remove "Column1"
baseline_df = baseline_df[:,2:end]
timedepend_df = timedepend_df[:,2:end]

other_vars = ["patient_id", "months_since_1st_test", "feeding_tube", 
            "scoliosis_yn", "pain_yn", "fatigue_yn", "ventilation", "adverse_event", 
            "fvc_yn", "fvc_percent",
            "gen_impr", "mf_impr", "rf_impr"
]

baseline_vars = names(baseline_df)[findall(x -> !(x ∈ ["cohort", "baseline_date"]), names(baseline_df))]
#baseline_vars = names(baseline_df)[findall(x -> !(x ∈ ["cohort", "baseline_date", "birth_d", "drug_sma"]), names(baseline_df))]

#------------------------------
# get data for specific tests
#------------------------------

test1="hfmse"
test2="rulm"
mixeddata = get_data_tests(timedepend_df, baseline_df, other_vars, baseline_vars; test1=test1, test2=test2, remove_lessthan1=true);

#------------------------------
# train jointly from scratch  
#------------------------------

nODEparams = 6
dynamics = params_fullinhomogeneous

# init models
modelargs1 = ModelArgs(p=size(mixeddata.xs1[1],1), 
                    q=length(mixeddata.xs_baseline[1]), 
                    dynamics=dynamics,
                    bottleneck=false,
                    seed=1234,
                    scale_sigmoid=2, 
                    add_diagonal=true 
)
modelargs2 = ModelArgs(p=size(mixeddata.xs2[1],1), 
                    q=length(mixeddata.xs_baseline[1]), 
                    dynamics=dynamics,
                    bottleneck=false,
                    seed=1234,
                    scale_sigmoid=2, 
                    add_diagonal=true 
)

#----------------------------------------
# neither penalty 
m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

# loss args
args_joint=LossArgs(
    λ_μpenalty=0.0f0,
    λ_adversarialpenalty=0.0f0,
    λ_variancepenalty=5.0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)
# prepare training
trainingargs_joint=TrainingArgs(warmup=false, epochs=10, lr=0.00003)
# train 
train_mixed_model!(m1, m2, mixeddata, args_joint, trainingargs_joint, verbose=false, plotting=false)
# save trained models 
no_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# ODE, no adversarial 
m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

# loss args
args_joint=LossArgs(
    λ_μpenalty=5.0f0,# 1.0f0
    λ_adversarialpenalty=0.0f0,#1.0f0,
    λ_variancepenalty=5.0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)
# prepare training
trainingargs_joint=TrainingArgs(warmup=false, epochs=10, lr=0.00003)# lr=0.00008
# train 
train_mixed_model!(m1, m2, mixeddata, args_joint, trainingargs_joint, verbose=false, plotting=false)

only_ODE_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# adversarial, no ODE 
m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

# loss args
args_joint=LossArgs(
    λ_μpenalty=0.0f0,# 1.0f0
    λ_adversarialpenalty=5.0f0,#1.0f0,
    λ_variancepenalty=5.0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)
# prepare training
trainingargs_joint=TrainingArgs(warmup=false, epochs=10, lr=0.00003)# lr=0.00008
# train 
train_mixed_model!(m1, m2, mixeddata, args_joint, trainingargs_joint, verbose=false, plotting=false)

only_adversarial_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# both adversarial and ODE 
m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

# loss args
args_joint=LossArgs(
    λ_μpenalty=5.0f0,# 1.0f0
    λ_adversarialpenalty=5.0f0,#1.0f0,
    λ_variancepenalty=5.0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)
# prepare training
trainingargs_joint=TrainingArgs(warmup=false, epochs=10, lr=0.00003)# lr=0.00008
# train 
train_mixed_model!(m1, m2, mixeddata, args_joint, trainingargs_joint, verbose=false, plotting=false)

ODE_and_adversarial_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# evaluation

# make corresponding directory to save results as CSV files 
save_eval_dir = "../results/penalties"
!isdir(save_eval_dir) && mkdir(save_eval_dir)

all_prederrs_df = DataFrame(
    PenaltyType = [],
    Dimension = [],
    ODE = [],
    LinearModel = []
)

# no penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(no_penalty_models["m1"], no_penalty_models["m2"], mixeddata, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
# baseline linear model
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false)
# append to overall df 
prederrdf[!, :PenaltyType] .= "No penalty"
append!(all_prederrs_df, prederrdf)

# only ODE penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(only_ODE_penalty_models["m1"], only_ODE_penalty_models["m2"], mixeddata, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false)
prederrdf[!, :PenaltyType] .= "Only ODE penalty"
append!(all_prederrs_df, prederrdf)

# only adversarial penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(only_adversarial_penalty_models["m1"], only_adversarial_penalty_models["m2"], mixeddata, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false)
prederrdf[!, :PenaltyType] .= "Only adversarial penalty"
append!(all_prederrs_df, prederrdf)

# both ODE and adversarial penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(ODE_and_adversarial_penalty_models["m1"], ODE_and_adversarial_penalty_models["m2"], mixeddata, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false)
prederrdf[!, :PenaltyType] .= "ODE and adversarial penalty"
append!(all_prederrs_df, prederrdf)

# save overall results to CSV 
CSV.write(joinpath(save_eval_dir, "prediction_errors.csv"), all_prederrs_df)

#----------------------------------------
# save plots in different versions 

# get selection of individuals 
Random.seed!(789)
selected_ids = rand(mixeddata.ids, 12)

# make corresponding directory
save_plots_dir = save_eval_dir
!isdir(save_plots_dir) && mkdir(save_plots_dir)

#----------------------------------------
# no penalties
plot_no_penalty = plot(
    plot_selected_ids_final(
        no_penalty_models["m1"], no_penalty_models["m2"], 
        mixeddata, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], 
        marker_sizes = [7, 5]
    ), 
    plot_title="No ODE or adversarial penalty"
)
savefig(plot_no_penalty, joinpath(save_plots_dir, "no_penalty.pdf"))

#----------------------------------------
# only ODE penalty
plot_only_ODE_penalty = plot(
    plot_selected_ids_final(
        only_ODE_penalty_models["m1"], only_ODE_penalty_models["m2"], 
        mixeddata, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], 
        marker_sizes = [7, 5]
    ), 
    plot_title="Only ODE penalty"
)
savefig(plot_only_ODE_penalty, joinpath(save_plots_dir, "only_ODE_penalty.pdf"))

#----------------------------------------
# only adversarial penalty

plot_only_adversarial_penalty = plot(
    plot_selected_ids_final(
        only_adversarial_penalty_models["m1"], only_adversarial_penalty_models["m2"], 
        mixeddata, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], 
        marker_sizes = [7, 5]
    ), 
    plot_title="Only adversarial penalty"
)
savefig(plot_only_adversarial_penalty, joinpath(save_plots_dir, "only_adversarial_penalty.pdf"))

#----------------------------------------
# both ODE and adversarial penalty

plot_ODE_and_adversarial_penalty = plot(
    plot_selected_ids_final(
        ODE_and_adversarial_penalty_models["m1"], ODE_and_adversarial_penalty_models["m2"], 
        mixeddata, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], 
        marker_sizes = [7, 5]
    ), 
    plot_title="ODE and adversarial penalty"
)
savefig(plot_ODE_and_adversarial_penalty, joinpath(save_plots_dir, "ODE_and_adversarial_penalty.pdf"))