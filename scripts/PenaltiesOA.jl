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

# load data with all medications
baseline_total_df = CSV.File(string(data_path, "baseline_df_total.csv"), truestrings = ["TRUE", "M"], falsestrings = ["FALSE", "F"], missingstring = ["NA"], decimal=',') |> DataFrame
timedepend_total_df = CSV.File(string(data_path, "timedepend_df_total.csv"), truestrings = ["TRUE"], falsestrings = ["FALSE"], missingstring = ["NA"], decimal=',') |> DataFrame
# remove column1
baseline_total_df = baseline_total_df[:,2:end]
timedepend_total_df = timedepend_total_df[:,2:end]
baseline_total_vars = names(baseline_total_df)[findall(x -> !(x ∈ ["cohort", "baseline_date", "birth_d", "drug_sma"]), names(baseline_total_df))]
other_vars = ["patient_id", "months_since_1st_test"]

# preprocess baseline_total_df 
# replace missings in mutation by false
baseline_total_df[ismissing.(baseline_total_df[:,:mutation_exon]), :mutation_exon] .= false
baseline_total_df[ismissing.(baseline_total_df[:,:mutation_compound]), :mutation_compound] .= false
baseline_total_df[ismissing.(baseline_total_df[:,:mutation_other]), :mutation_other] .= false
baseline_total_df[ismissing.(baseline_total_df[:,:presym_diag]), :presym_diag] .= false
# calculate age based on birth date and baseline data, assuming that the first treatment was at baseline
rows_with_missing_age = findall(x -> ismissing(x), baseline_total_df[:,:age_at_first_treatment])
imputed_ages_days = baseline_total_df[rows_with_missing_age,:baseline_date] .- baseline_total_df[rows_with_missing_age,:birth_d]
imputed_ages_months = Int.(round.(Dates.value.(imputed_ages_days) ./ 30.5))
baseline_total_df[rows_with_missing_age,:age_at_first_treatment] = imputed_ages_months

cohorts = unique(baseline_total_df[:,:cohort])
therapies = sort(unique(baseline_total_df[:,:drug_sma]))

# filter for Zolgensma only patients
zolg_therapies = ["Monotherapie Zolgensma", therapies[findall(x -> startswith(x, "zolg"), lowercase.(therapies))]...]
zolgensma_baseline_df = filter(x -> x.drug_sma ∈ zolg_therapies, baseline_total_df)
zolgensma_ids = unique(zolgensma_baseline_df[:,:patient_id])
zolgensma_timedepend_df = filter(x -> x.patient_id ∈ zolgensma_ids, timedepend_total_df)

#------------------------------
# get data for specific tests
#------------------------------

test1="hfmse"
test2="rulm"

mixeddata_zolg = get_SMArtCARE_data_one_test(zolgensma_timedepend_df, zolgensma_baseline_df, other_vars, baseline_vars; test1=test1, test2=test2, remove_lessthan1=true); # 76
n_patients_zolg = length(mixeddata_zolg.ids)

#----------------------------------------
# ZOLGENSMA: train jointly from scratch
#----------------------------------------

nODEparams = 2 
dynamics = params_diagonalhomogeneous

# init models
modelargs1 = ModelArgs(p=size(mixeddata_zolg.xs1[1],1), 
                    q=length(mixeddata_zolg.xs_baseline[1]), 
                    dynamics=dynamics,
                    bottleneck=false,
                    seed=1234,
                    scale_sigmoid=2, # structure of the baseline network: controls the range of the ODE parameters
                    add_diagonal=true # structure of the baseline network: whether or not a diagonal transformation is added
)
modelargs2 = ModelArgs(p=size(mixeddata_zolg.xs2[1],1), 
                    q=length(mixeddata_zolg.xs_baseline[1]), # zdim=2 default
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

args_joint=LossArgs(
    λ_μpenalty=0.0f0,
    λ_adversarialpenalty=0.0f0,
    λ_variancepenalty=2.0f0,
    variancepenaltytype = :log_diff,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(warmup=false, epochs=15, lr=0.001)
train_mixed_model!(m1, m2, mixeddata_zolg, args_joint, trainingargs_joint, verbose=true, plotting=true)

no_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# ODE, no adversarial 

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

args_joint=LossArgs(
    λ_μpenalty=5.0f0,
    λ_adversarialpenalty=0.0f0,
    λ_variancepenalty=2.0f0,
    variancepenaltytype = :log_diff,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(warmup=false, epochs=15, lr=0.001)
train_mixed_model!(m1, m2, mixeddata_zolg, args_joint, trainingargs_joint, verbose=true, plotting=true)

only_ODE_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# adversarial, no ODE 

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

args_joint=LossArgs(
    λ_μpenalty=0.0f0,
    λ_adversarialpenalty=5.0f0,
    λ_variancepenalty=2.0f0,
    variancepenaltytype = :log_diff,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(warmup=false, epochs=15, lr=0.001)
train_mixed_model!(m1, m2, mixeddata_zolg, args_joint, trainingargs_joint, verbose=true, plotting=true)

only_adversarial_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# both adversarial and ODE 

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

args_joint=LossArgs(
    λ_μpenalty=5.0f0,
    λ_adversarialpenalty=5.0f0,
    λ_variancepenalty=2.0f0,
    variancepenaltytype = :log_diff,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(warmup=false, epochs=15, lr=0.001)
train_mixed_model!(m1, m2, mixeddata_zolg, args_joint, trainingargs_joint, verbose=true, plotting=true)

ODE_and_adversarial_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# evaluation

# make corresponding directory to save results as CSV files 
save_eval_dir = "../results/zolgensma"
!isdir(save_eval_dir) && mkdir(save_eval_dir)

all_prederrs_df = DataFrame(
    PenaltyType = [],
    Dimension = [],
    ODE = [],
    LinearModel = []
)

# no penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(no_penalty_models["m1"], no_penalty_models["m2"], mixeddata_zolg, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
# baseline linear model
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false, dataset="zolgensma")
# append to overall df 
prederrdf[!, :PenaltyType] .= "No penalty"
append!(all_prederrs_df, prederrdf)

# only ODE penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(only_ODE_penalty_models["m1"], only_ODE_penalty_models["m2"], mixeddata_zolg, args_joint, 
        verbose=false
)
#
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false, dataset="zolgensma")
prederrdf[!, :PenaltyType] .= "Only ODE penalty"
append!(all_prederrs_df, prederrdf)

# only adversarial penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(only_adversarial_penalty_models["m1"], only_adversarial_penalty_models["m2"], mixeddata_zolg, args_joint, 
        verbose=false
)
#
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false, dataset="zolgensma")
prederrdf[!, :PenaltyType] .= "Only adversarial penalty"
append!(all_prederrs_df, prederrdf)

# both ODE and adversarial penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(ODE_and_adversarial_penalty_models["m1"], ODE_and_adversarial_penalty_models["m2"], mixeddata_zolg, args_joint, 
        verbose=false
)
#
prederrdf = fit_baseline_model(df_all, modelargs1.q; verbose=false, dataset="zolgensma")
prederrdf[!, :PenaltyType] .= "ODE and adversarial penalty"
append!(all_prederrs_df, prederrdf)

# save overall results to CSV 
CSV.write(joinpath(save_eval_dir, "prediction_errors.csv"), all_prederrs_df)

#----------------------------------------
# save plots in different versions 

# get selection of individuals 
Random.seed!(786)
selected_ids = rand(mixeddata_zolg.ids, 12)

# make corresponding directory
save_plots_dir = save_eval_dir #"Results/plots_penalties_risdiplam"
!isdir(save_plots_dir) && mkdir(save_plots_dir)

#----------------------------------------
# no penalties
plot_no_penalty = plot(
    plot_selected_ids(no_penalty_models["m1"], no_penalty_models["m2"], 
        mixeddata_zolg, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="No ODE or adversarial penalty"
)
savefig(plot_no_penalty, joinpath(save_plots_dir, "no_penalty.pdf"))

#----------------------------------------
# only ODE penalty
plot_only_ODE_penalty = plot(
    plot_selected_ids(only_ODE_penalty_models["m1"], only_ODE_penalty_models["m2"], 
        mixeddata_zolg, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="Only ODE penalty"
)
savefig(plot_only_ODE_penalty, joinpath(save_plots_dir, "only_ODE_penalty.pdf"))

#----------------------------------------
# only adversarial penalty
plot_only_adversarial_penalty = plot(
    plot_selected_ids(only_adversarial_penalty_models["m1"], only_adversarial_penalty_models["m2"], 
        mixeddata_zolg, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="Only adversarial penalty"
)
savefig(plot_only_adversarial_penalty, joinpath(save_plots_dir, "only_adversarial_penalty.pdf"))

#----------------------------------------
# both ODE and adversarial penalty
plot_ODE_and_adversarial_penalty = plot(
    plot_selected_ids(ODE_and_adversarial_penalty_models["m1"], ODE_and_adversarial_penalty_models["m2"], 
        mixeddata_zolg, args_joint, selected_ids, 
        colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="ODE and adversarial penalty"
)
savefig(plot_ODE_and_adversarial_penalty, joinpath(save_plots_dir, "ODE_and_adversarial_penalty.pdf"))