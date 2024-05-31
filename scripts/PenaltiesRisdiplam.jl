#------------------------------
# setup
#------------------------------

cd(@__DIR__)

using Pkg;
Pkg.activate(".")
Pkg.status()

using CSV
using Dates
using DataFrames
using Distributions
using Random
using Flux
using Flux: DataLoader
using LaTeXStrings
using LinearAlgebra
using Measures
using MixedModels
using Parameters
using Plots 
using ProgressMeter
using StatsBase

gr()

includet("PreprocessingSMArtCARE.jl")
includet("load_data.jl")
includet("model.jl")
includet("training.jl")
includet("ODEsolutions.jl")
includet("PlottingNew.jl")
includet("plotting_SMA_mixed.jl")
includet("evaluate_mixed.jl")
includet("write_config.jl")

#------------------------------
# load data
#------------------------------

# new dataset
data_path = joinpath("newdataset/")

# load total data 
baseline_total_df = CSV.File(string(data_path, "baseline_df_total.csv"), truestrings = ["TRUE", "M"], falsestrings = ["FALSE", "F"], missingstring = ["NA"], decimal=',') |> DataFrame
timedepend_total_df = CSV.File(string(data_path, "timedepend_df_total.csv"), truestrings = ["TRUE"], falsestrings = ["FALSE"], missingstring = ["NA"], decimal=',') |> DataFrame

# remove Column1
baseline_total_df = baseline_total_df[:,2:end]
timedepend_total_df = timedepend_total_df[:,2:end]

other_vars = ["patient_id", "months_since_1st_test"]
baseline_total_vars = names(baseline_total_df)[findall(x -> !(x ∈ ["cohort", "baseline_date", "birth_d", "drug_sma"]), names(baseline_total_df))]

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

# filter for Risdiplam patients 
#risdi_baseline_df = filter(x -> x.drug_sma ∈ ["Monotherapie Risdiplam", "Risdi_Zolg_switcher", "Risdi_Nusi_switcher"], baseline_total_df)
risdi_baseline_df = filter(x -> x.drug_sma ∈ ["Monotherapie Risdiplam"], baseline_total_df)
risdi_ids = unique(risdi_baseline_df[:,:patient_id])
risdi_timedepend_df = filter(x -> x.patient_id ∈ risdi_ids, timedepend_total_df)

#------------------------------
# get data for specific tests
#------------------------------

test1="hfmse"
test2="rulm"

mixeddata_risdi = get_data_tests(risdi_timedepend_df, risdi_baseline_df, other_vars, baseline_total_vars; test1=test1, test2=test2, remove_lessthan1=true); # 154
n_patients_risdi = length(mixeddata_risdi.ids)


#------------------------------
# RISDIPLAM: train jointly from scratch  
#------------------------------

nODEparams = 6
dynamics = params_fullinhomogeneous # Matrix ohne Nullen + konstanter Term  

#nODEparams = 4
#dynamics = params_fullhomogeneous
#dynamics = params_diagonalinhomogeneous

#nODEparams = 2 
#dynamics = params_diagonalhomogeneous

# init models
modelargs1 = ModelArgs(p=size(mixeddata_risdi.xs1[1],1), 
                    q=length(mixeddata_risdi.xs_baseline[1]), # zdim=2 default
                    dynamics=dynamics,
                    bottleneck=false,
                    seed=1234,
                    scale_sigmoid=2, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                    add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
)
modelargs2 = ModelArgs(p=size(mixeddata_risdi.xs2[1],1), 
                    q=length(mixeddata_risdi.xs_baseline[1]), # zdim=2 default
                    dynamics=dynamics,
                    bottleneck=false,
                    seed=1234,
                    scale_sigmoid=2, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                    add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
)

#----------------------------------------
# neither penalty 

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

args_joint=LossArgs(
    λ_μpenalty=0.0f0,
    λ_adversarialpenalty=0.0f0,
    λ_variancepenalty=5.0f0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(epochs=45, lr=0.0001, warmup=false)
train_mixed_model!(m1, m2, mixeddata_risdi, args_joint, trainingargs_joint, verbose=true, plotting=true)

no_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# ODE, no adversarial 

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

# loss args
args_joint=LossArgs(
    λ_μpenalty=5.0f0,
    λ_adversarialpenalty=0.0f0,
    λ_variancepenalty=5.0f0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(epochs=45, lr=0.0001, warmup=false)
train_mixed_model!(m1, m2, mixeddata_risdi, args_joint, trainingargs_joint, verbose=true, plotting=true)

only_ODE_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# adversarial, no ODE 

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

# loss args
args_joint=LossArgs(
    λ_μpenalty=0.0f0,
    λ_adversarialpenalty=5.0f0,
    λ_variancepenalty=5.0f0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(epochs=45, lr=0.0001, warmup=false)
train_mixed_model!(m1, m2, mixeddata_risdi, args_joint, trainingargs_joint, verbose=true, plotting=true)

only_adversarial_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# both adversarial and ODE 

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

# loss args
args_joint=LossArgs(
    λ_μpenalty=5.0f0,# 1.0f0
    λ_adversarialpenalty=5.0f0,#1.0f0,
    λ_variancepenalty=5.0f0,
    variancepenaltytype = :sum_ratio,
    variancepenaltyoffset = 1.0f0, 
    skipt0=true,
    weighting=true, 
)

trainingargs_joint=TrainingArgs(epochs=45, lr=0.0001, warmup=false)
train_mixed_model!(m1, m2, mixeddata_risdi, args_joint, trainingargs_joint, verbose=true, plotting=true)

ODE_and_adversarial_penalty_models = Dict("m1" => deepcopy(m1), "m2" => deepcopy(m2))

#----------------------------------------
# evaluation

# variable 23 "onset_prenatal" is excluded from the linear model because it is nearly all zero (sum=1)

# make corresponding directory to save results as CSV files 
save_eval_dir = "Results/risdiplam_6params"
!isdir(save_eval_dir) && mkdir(save_eval_dir)

all_prederrs_df = DataFrame(
    PenaltyType = [],
    Dimension = [],
    ODE = [],
    FullMixedModel = [],
    SimpleMixedModel = [],
    LinearModel = []
)

# no penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(no_penalty_models["m1"], no_penalty_models["m2"], mixeddata_risdi, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
# baseline mixed and linear models
prederrdf = fit_baseline_models(df_all, modelargs1.q; verbose=false, dataset="risdiplam")
# append to overall df 
prederrdf[!, :PenaltyType] .= "No penalty"
append!(all_prederrs_df, prederrdf)

# only ODE penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(only_ODE_penalty_models["m1"], only_ODE_penalty_models["m2"], mixeddata_risdi, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
prederrdf = fit_baseline_models(df_all, modelargs1.q; verbose=false, dataset="risdiplam")
prederrdf[!, :PenaltyType] .= "Only ODE penalty"
append!(all_prederrs_df, prederrdf)

# only adversarial penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(only_adversarial_penalty_models["m1"], only_adversarial_penalty_models["m2"], mixeddata_risdi, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
prederrdf = fit_baseline_models(df_all, modelargs1.q; verbose=false, dataset="risdiplam")
prederrdf[!, :PenaltyType] .= "Only adversarial penalty"
append!(all_prederrs_df, prederrdf)

# both ODE and adversarial penalty
df_all, ODEprederrs1, ODEprederrs2 = 
    eval_prediction(ODE_and_adversarial_penalty_models["m1"], ODE_and_adversarial_penalty_models["m2"], mixeddata_risdi, args_joint, 
        verbose=false
)
#
mean(ODEprederrs1[.!isnan.(ODEprederrs1)])
mean(ODEprederrs2[.!isnan.(ODEprederrs2)])
prederrdf = fit_baseline_models(df_all, modelargs1.q; verbose=false, dataset="risdiplam")
prederrdf[!, :PenaltyType] .= "ODE and adversarial penalty"
append!(all_prederrs_df, prederrdf)

# save overall results to CSV 
CSV.write(joinpath(save_eval_dir, "prediction_errors.csv"), all_prederrs_df)

#----------------------------------------
# save plots in different versions 

# get selection of individuals 
Random.seed!(5789) # 789
selected_ids = rand(mixeddata_risdi.ids, 12)

# make corresponding directory
save_plots_dir = save_eval_dir #"Results/plots_penalties_risdiplam"
!isdir(save_plots_dir) && mkdir(save_plots_dir)

#----------------------------------------
# no penalties
plot_no_penalty = plot(plot_selected_ids_final(no_penalty_models["m1"], no_penalty_models["m2"], 
    mixeddata_risdi, args_joint, selected_ids, 
    colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="No ODE or adversarial penalty"
)
savefig(plot_no_penalty, joinpath(save_plots_dir, "no_penalty.pdf"))

#----------------------------------------
# only ODE penalty
plot_only_ODE_penalty = plot(plot_selected_ids_final(only_ODE_penalty_models["m1"], only_ODE_penalty_models["m2"], 
    mixeddata_risdi, args_joint, selected_ids, 
    colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="Only ODE penalty"
)
savefig(plot_only_ODE_penalty, joinpath(save_plots_dir, "only_ODE_penalty.pdf"))

#----------------------------------------
# only adversarial penalty
plot_only_adversarial_penalty = plot(plot_selected_ids_final(only_adversarial_penalty_models["m1"], only_adversarial_penalty_models["m2"], 
    mixeddata_risdi, args_joint, selected_ids, 
    colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="Only adversarial penalty"
)
savefig(plot_only_adversarial_penalty, joinpath(save_plots_dir, "only_adversarial_penalty.pdf"))

#----------------------------------------
# both ODE and adversarial penalty
plot_ODE_and_adversarial_penalty = plot(plot_selected_ids_final(ODE_and_adversarial_penalty_models["m1"], ODE_and_adversarial_penalty_models["m2"], 
    mixeddata_risdi, args_joint, selected_ids, 
    colors_points = ["#3182bd" "#9ecae1"; "#e6550d" "#fdae6b"], marker_sizes = [7, 5]), 
    plot_title="ODE and adversarial penalty"
)
savefig(plot_ODE_and_adversarial_penalty, joinpath(save_plots_dir, "ODE_and_adversarial_penalty.pdf"))

#----------------------------------------
# evaluation with "Fig 2/3-like" plots from the paper

includet("functions_modifications.jl")

# no penalties 
agg_abs_delta_df = make_df_from_deltas(no_penalty_models["m1"], no_penalty_models["m2"], mixeddata_risdi)
deltaplot = create_delta_scatterplots(agg_abs_delta_df, 
    fix_limits=false, saveplot=true, savepath=save_plots_dir, filename="deltas_scatter_nopenalties.pdf"
)
above_diagonal_df = collect_stats_about_deltas(agg_abs_delta_df)
CSV.write(joinpath(save_eval_dir, "above_diagonal_nopenalties.csv"), above_diagonal_df)

# only ODE penalty
agg_abs_delta_df = make_df_from_deltas(only_ODE_penalty_models["m1"], only_ODE_penalty_models["m2"], mixeddata_risdi)
deltaplot = create_delta_scatterplots(agg_abs_delta_df, 
    fix_limits=false, saveplot=true, savepath=save_plots_dir, filename="deltas_scatter_onlyODEpenalty.pdf"
)
above_diagonal_df = collect_stats_about_deltas(agg_abs_delta_df)
CSV.write(joinpath(save_eval_dir, "above_diagonal_onlyODEpenalty.csv"), above_diagonal_df)

# only adversarial penalty
agg_abs_delta_df = make_df_from_deltas(only_adversarial_penalty_models["m1"], only_adversarial_penalty_models["m2"], mixeddata_risdi)
deltaplot = create_delta_scatterplots(agg_abs_delta_df, 
    fix_limits=false, saveplot=true, savepath=save_plots_dir, filename="deltas_scatter_onlyadversarialpenalty.pdf"
)
above_diagonal_df = collect_stats_about_deltas(agg_abs_delta_df)
CSV.write(joinpath(save_eval_dir, "above_diagonal_onlyadversarialpenalty.csv"), above_diagonal_df)

# both penalties
agg_abs_delta_df = make_df_from_deltas(ODE_and_adversarial_penalty_models["m1"], ODE_and_adversarial_penalty_models["m2"], mixeddata_risdi)
deltaplot = create_delta_scatterplots(agg_abs_delta_df, 
    fix_limits=false, saveplot=true, savepath=save_plots_dir, filename="deltas_scatter_bothpenalties.pdf"
)
above_diagonal_df = collect_stats_about_deltas(agg_abs_delta_df)
CSV.write(joinpath(save_eval_dir, "above_diagonal_bothpenalties.csv"), above_diagonal_df)