#------------------------------
# Setup
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
include(joinpath(sourcedir, "modify_data.jl"))
include(joinpath(sourcedir, "model.jl"))
include(joinpath(sourcedir, "ODE_solutions.jl"))
include(joinpath(sourcedir, "training.jl"))
include(joinpath(sourcedir, "training_modifications.jl"))
include(joinpath(sourcedir, "eval_modifications.jl"))
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

# preprocess data for a specific test 
test = "rulm"
testdata, sumscores, keep_timepoint_masks = get_SMArtCARE_data(test, baseline_df, timedepend_df, extended_output=true);
# recode items 
recoded_testdata = recode_SMArtCARE_data(testdata);

subscale2_letters = ["j", "k", "l", "m", "n"]
subscale2_names = collect("rulm_item$(letter)" for letter in subscale2_letters)
subscale2_inds = [10, 11, 12, 13, 14]
testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
#xs2, xs_baseline2, tvals2, ids2 = testdata2.xs, testdata2.xs_baseline, testdata2.tvals, testdata2.ids;
recoded_testdata2 = recode_SMArtCARE_data(testdata2);

#------------------------------
# modification settings
#------------------------------

p_dropout = 0.5
shrink_prob_offset = 3
shift = 2
p_subgroup = 0.5

# look at distribution of sum scores
sumscores_arr = cat(sumscores..., dims=1)
sumscore_cutoff = Int(quantile(sumscores_arr, 0.60))

parentdir = "../results/modifications/"
!isdir(parentdir) && mkdir(parentdir)

#------------------------------
# model and training settings
#------------------------------

# set dynamics
dynamics = params_fullhomogeneous 

# define numbers of individuals and variables 
n = length(recoded_testdata.xs)

penaltytype = :sum_ratio
penaltyweight = 5.0f0
lr = 0.001
penaltyoffset = 1.0f0
epoch_checkpoints = [10, 15, 20, 25, 30]

# loss args
args_joint=LossArgs(
    λ_adversarialpenalty=5.0f0,
    skipt0=true, 
    weighting=true, 
    λ_variancepenalty = penaltyweight,
    variancepenaltytype = penaltytype,
    variancepenaltyoffset = penaltyoffset, 
    λ_μpenalty=5.0f0
)

#########################################################################################
#########################################################################################
#------------------------------
# Modification 1-6
#------------------------------

for mod_no in 1:6
    @info "Modification $(mod_no)..."

    recoded_testdata2, pathname = modify_data(
        test, baseline_df, timedepend_df, subscale2_names, sumscores, 
        mod_no, 
        p_dropout, shrink_prob_offset, shift, p_subgroup, sumscore_cutoff, 
        parentdir
    );

    mixeddata = SMAMixedTestData("rulm_handle_weights", "rulm", 
                                recoded_testdata2.xs, recoded_testdata.xs, 
                                recoded_testdata.xs_baseline,
                                recoded_testdata2.tvals, recoded_testdata.tvals, 
                                recoded_testdata2.ids, recoded_testdata.ids, recoded_testdata.ids
    );

    # train jointly from scratch  

    # define model 
    modelargs1 = ModelArgs(p=size(mixeddata.xs1[1],1), 
                        q=length(mixeddata.xs_baseline[1]), # zdim=2 default
                        dynamics=dynamics,
                        seed=1234, #1234
                        bottleneck=false, #true, #false for 2 param systems
                        scale_sigmoid=2.0f0, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                        add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
    )

    modelargs2 = ModelArgs(p=size(mixeddata.xs2[1],1), 
                        q=length(mixeddata.xs_baseline[1]), # zdim=2 default
                        #nODEparams=nODEparams,
                        dynamics=dynamics,
                        bottleneck=false,
                        seed=1234,
                        scale_sigmoid=2, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                        add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
    )

    #resultspath = "Results/rulm_small_subscale_modification_$(mod_no)"#"Results/rulm_small_subscale"
    # for training without modifications: pathname = joinpath(parentdir, "no_modification")
    resultspath = pathname
    !isdir(resultspath) && mkdir(resultspath)

    if mod_no == 6 # reduce variance penalty 
        penaltyweight = 2.0f0
        args_joint=LossArgs(
            λ_variancepenalty = penaltyweight,
            variancepenaltytype = penaltytype,
            variancepenaltyoffset = penaltyoffset,
            λ_adversarialpenalty=5.0f0, λ_μpenalty=5.0f0,
            skipt0=true, weighting=true
        )
    end
    @info "Penalty weight: $(penaltyweight)"

    # start training
    Random.seed!(48)

    #tmp_description = "adv$(args_joint.λ_adversarialpenalty)_var_$(penaltytype)_weight$(penaltyweight)_offset$(penaltyoffset)_μ_$(args_joint.λ_μpenalty)_lr$(lr)"
    #configpath = check_and_create_config_path(resultspath, tmp_description)

    m1 = odevae(modelargs1);
    m2 = odevae(modelargs2);

    do_training_checkpoints(m1, m2, mixeddata, args_joint, epoch_checkpoints, lr, 
        resultspath, modelargs1, modelargs2,
        save_individual_plots=true, save_delta_stats=true, fix_axis_limits=false)
        #save_individual_plots=false, save_delta_stats=false, fix_axis_limits=true)
    #do_training_checkpoints(m1, m2, mixeddata, args_joint, 30, lr, configpath)
end

#########################################################################################
#########################################################################################
#------------------------------
# No modification 
#------------------------------

# preprocess data again 
test = "rulm"
testdata, sumscores, keep_timepoint_masks = get_SMArtCARE_data(test, baseline_df, timedepend_df, extended_output=true);
recoded_testdata = recode_SMArtCARE_data(testdata);
#xs_names = names(timedepend_df)[9:28]

subscale2_letters = ["j", "k", "l", "m", "n"]
subscale2_names = collect("rulm_item$(letter)" for letter in subscale2_letters)
subscale2_inds = [10, 11, 12, 13, 14]
testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
recoded_testdata2 = recode_SMArtCARE_data(testdata2);

mixeddata = SMAMixedTestData("rulm_handle_weights", "rulm", 
                            recoded_testdata2.xs, recoded_testdata.xs, 
                            recoded_testdata.xs_baseline,
                            recoded_testdata2.tvals, recoded_testdata.tvals, 
                            recoded_testdata2.ids, recoded_testdata2.ids, recoded_testdata2.ids
);

# train jointly from scratch  

# define model 
modelargs1 = ModelArgs(p=size(mixeddata.xs1[1],1), 
                    q=length(mixeddata.xs_baseline[1]), # zdim=2 default
                    dynamics=dynamics,
                    seed=1234, #1234
                    bottleneck=false, #true, #false for 2 param systems
                    scale_sigmoid=2.0f0, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                    add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
)

modelargs2 = ModelArgs(p=size(mixeddata.xs2[1],1), 
                    q=length(mixeddata.xs_baseline[1]), # zdim=2 default
                    #nODEparams=nODEparams,
                    dynamics=dynamics,
                    bottleneck=false,
                    seed=1234,
                    scale_sigmoid=2, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                    add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
)

pathname = joinpath(parentdir, "no_modification")
resultspath = pathname
!isdir(resultspath) && mkdir(resultspath)

no_mod_epoch_checkpoints = [2, 5, 10, 12, 15, 18, 20, 25, 30]
penaltyweight = 5.0f0

args_joint=LossArgs(
    λ_adversarialpenalty=5.0f0,
    skipt0=true, 
    weighting=true, 
    λ_variancepenalty = penaltyweight,
    variancepenaltytype = penaltytype,
    variancepenaltyoffset = penaltyoffset, 
    λ_μpenalty=5.0f0
)

# start training
Random.seed!(48)

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

do_training_checkpoints(m1, m2, mixeddata, args_joint, no_mod_epoch_checkpoints, lr, 
    resultspath, modelargs1, modelargs2,
    save_individual_plots=true, save_delta_stats=true, fix_axis_limits=false,
    #save_individual_plots=false, save_delta_stats=false,
    #fix_axis_limits=true, xlims=(0, 4.2), ylims=(0, 5.2)
)

#########################################################################################
#########################################################################################
#------------------------------
# Delete items of larger instrument 
#------------------------------

# delete items of first measurement instrument: 15, 16, 17, 18, 19, 20
# corresponding to shoulder abduction and shoulder flexion: o, p, q, r, s, t

subscale1_letters = string.([collect('a':'i')..., collect('o':'t')...])
subscale1_names = collect("rulm_item$(letter)" for letter in subscale1_letters)
testdata_sub = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale1_names);
recoded_testdata_sub = recode_SMArtCARE_data(testdata_sub);

subscale2_letters = ["j", "k", "l", "m", "n"]
subscale2_names = collect("rulm_item$(letter)" for letter in subscale2_letters)
testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
recoded_testdata2 = recode_SMArtCARE_data(testdata2);

mixeddata = SMAMixedTestData("rulm_handle_weights", "rulm_no_shoulder", 
                            recoded_testdata2.xs, recoded_testdata_sub.xs, 
                            recoded_testdata2.xs_baseline,
                            recoded_testdata2.tvals, recoded_testdata_sub.tvals, 
                            recoded_testdata2.ids, recoded_testdata2.ids, recoded_testdata2.ids
);

# train jointly from scratch  

# define model 
modelargs1 = ModelArgs(p=size(mixeddata.xs1[1],1), 
                    q=length(mixeddata.xs_baseline[1]), # zdim=2 default
                    dynamics=dynamics,
                    seed=1234, #1234
                    bottleneck=false, #true, #false for 2 param systems
                    scale_sigmoid=2.0f0, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                    add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
)

modelargs2 = ModelArgs(p=size(mixeddata.xs2[1],1), 
                    q=length(mixeddata.xs_baseline[1]), # zdim=2 default
                    #nODEparams=nODEparams,
                    dynamics=dynamics,
                    bottleneck=false,
                    seed=1234,
                    scale_sigmoid=2, # Struktur des ODE-net: zulässiger Bereich der ODE-Parameter
                    add_diagonal=true # Struktur des ODE-net: Diagonal-Layer am Ende Ja/Nein
)

pathname = joinpath(parentdir, "larger_test_without_subscale")
resultspath = pathname
!isdir(resultspath) && mkdir(resultspath)

epoch_checkpoints = [10, 15, 20, 25, 30]
penaltyweight = 5.0f0

args_joint=LossArgs(
    λ_adversarialpenalty=5.0f0,
    skipt0=true, 
    weighting=true, 
    λ_variancepenalty = penaltyweight,
    variancepenaltytype = penaltytype,
    variancepenaltyoffset = penaltyoffset, 
    λ_μpenalty=5.0f0
)

# start training
Random.seed!(48)

m1 = odevae(modelargs1);
m2 = odevae(modelargs2);

do_training_checkpoints(m1, m2, mixeddata, args_joint, epoch_checkpoints, lr, 
    resultspath, modelargs1, modelargs2,
    save_individual_plots=true, save_delta_stats=true, fix_axis_limits=false
    #save_individual_plots=false, save_delta_stats=false, fix_axis_limits=true
)

#########################################################################################
#########################################################################################

# collect delta stats

overall_delta_stats = DataFrame(
    dimension = String[], 
    no_above_diagonal = Int[],
    no_total = Int[],
    perc_above_diagonal = Float64[],
    modification = String[],
    epoch = Int[]
)

penaltyweight = 5.0f0
cur_delta_df = nothing

# Numbered modifications 1-6
for mod_no in 1:6
    @info mod_no
    recoded_testdata2, pathname = modify_data(
        test, baseline_df, timedepend_df, subscale2_names, sumscores, 
        mod_no, 
        p_dropout, shrink_prob_offset, shift, p_subgroup, sumscore_cutoff, 
        parentdir
    );
    resultspath = pathname
    modification_string = split(resultspath, "/")[end]
    mod_no == 6 && (penaltyweight = 2.0f0)
    curpath = resultspath
    for epoch in epoch_checkpoints
        #@info epoch 
        try
            cur_delta_df = CSV.read(joinpath(curpath, "above_diagonal_stats_$(epoch)epochs.csv"), DataFrame)
        catch
            continue
        end
        cur_delta_df.epoch .= epoch
        cur_delta_df.modification .= modification_string
        append!(overall_delta_stats, cur_delta_df)
    end
end

# Larger test without subscale
modification_string = "larger_test_without_subscale"
resultspath = joinpath(parentdir, modification_string)
penaltyweight = 5.0f0
curpath = resultspath
for epoch in epoch_checkpoints
    try
        cur_delta_df = CSV.read(joinpath(curpath, "above_diagonal_stats_$(epoch)epochs.csv"), DataFrame)
    catch 
        continue
    end
    cur_delta_df.epoch .= epoch
    cur_delta_df.modification .= modification_string
    append!(overall_delta_stats, cur_delta_df)
end

# No modification
modification_string = "no_modification"
resultspath = joinpath(parentdir, modification_string)
epoch_checkpoints = [2, 5, 10, 12, 15, 18, 20, 25, 30]
penaltyweight = 5.0f0
curpath = resultspath
for epoch in epoch_checkpoints
    try
        cur_delta_df = CSV.read(joinpath(curpath, "above_diagonal_stats_$(epoch)epochs.csv"), DataFrame)
    catch 
        continue
    end    
    cur_delta_df.epoch .= epoch
    cur_delta_df.modification .= modification_string
    append!(overall_delta_stats, cur_delta_df)
end

modifications_df = filter(x -> 
    (x.modification != "mod_5_sumscore_cutoff_25") && 
    (x.modification != "mod_6_sumscore_cutoff_25") && 
    (x.epoch == 30),
    overall_delta_stats
)

mod_6_df = filter(x -> 
    (x.modification == "mod_6_sumscore_cutoff_25") && 
    (x.epoch == 30),
    overall_delta_stats
)
append!(modifications_df, mod_6_df)
modifications_df.perc_above_diagonal = round.(modifications_df.perc_above_diagonal, digits=3)

modifications_df = select(modifications_df, [:modification, :dimension, :perc_above_diagonal])
modifications_df[!,:perc_above_diagonal] = round.(modifications_df[!,:perc_above_diagonal], digits=3)

long_mod_df = unstack(modifications_df, :modification, :perc_above_diagonal)
long_mod_df = select!(long_mod_df, ["dimension", "no_modification", 
                                    "mod_1_p_dropout_0.5", "mod_2_shrink_prob_offset_3", 
                                    "mod_3_shift_2", "mod_4_shift_2_p_subgroup_0.5", 
                                    "mod_6_sumscore_cutoff_25", "larger_test_without_subscale"]
)

long_mod_df = rename(long_mod_df, ["Dimension", 
    "1) No modification",
    "4) Random dropout", "5) Time-dependent dropout", 
    "2) Constant shift", "3) Shift in subgroup", 
    "6) State-dependent availability", "7) Non-overlapping subscales"]
)
long_mod_df = long_mod_df[:, [1, 2, 5, 6, 3, 4, 7, 8]]

CSV.write(joinpath(parentdir, "modifications_perc_above_diagonal_all.csv"), long_mod_df)

# Now filter and save dataframe for no modification results

no_mod_df = filter(x -> 
    (x.modification == "no_modification") && 
    (x.epoch ∈ [2, 5, 10, 15, 20, 25, 30]),
    overall_delta_stats
)
select!(no_mod_df, [:dimension, :perc_above_diagonal, :epoch])
no_mod_df.perc_above_diagonal = round.(no_mod_df.perc_above_diagonal, digits=3)
long_no_mod_df = unstack(no_mod_df, :epoch, :perc_above_diagonal)
long_no_mod_df = rename(long_no_mod_df, 
    ["Dimension", "2 epochs", "5 epochs", "10 epochs", "15 epochs", "20 epochs", "25 epochs", "30 epochs"]
)
CSV.write(joinpath(parentdir, "no_modification_perc_above_diagonal.csv"), long_no_mod_df)