"""
This script reproduces Figures 2 and 3  in Section 3.3 and Tables 4 and 5 in Section A.4. 

The modifications are referred to by number in this script. 
To see which modification corresponds to which number, 
please refer to either the docstring of the function `modify_data` 
in `modify_data.jl`, or the following list. 

Only modifications 1-4 and 6 are considered in the final results.

- `mod_no = 1`: delete items at random time points (see `delete_at_random`)
- `mod_no = 2`: delete items at later time points preferentially (see `delete_at_later_tps`)
- `mod_no = 3`: apply a shift for all patients and items (see `shift_all_patients_items`)
- `mod_no = 4`: apply a shift for a random subgroup of patients (see `shift_random_subgrop`)
- `mod_no = 5`: delete items at later time points if the sum score of the other test 
            is above a certain threshold (see `delete_later_above_threshold`)
- `mod_no = 6`: delete items at earlier time points if the sum score of the other test 
            is above a certain threshold (see `delete_earlier_above_threshold`)
"""

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

USE_DUMMY_DATA = true

if USE_DUMMY_DATA
    !isdir("../results/dummy") && mkdir("../results/dummy")
    # read data
    baseline_df = CSV.File(string(data_path, "dummy_baseline_df.csv")) |> DataFrame
    timedepend_df = CSV.File(string(data_path, "dummy_timedepend_df.csv")) |> DataFrame
    # preprocess
    test = testname = "test1"
    recoded_testdata = get_dummy_data_one_test(testname, baseline_df, timedepend_df)
    sumscores = [vec(sum(recoded_testdata.xs[i],dims=1)) for i in 1:length(recoded_testdata.xs[:,1])]
    # subscale for smaller second instrument
    subscale2_names = collect("test1_item$(i)" for i in 11:15)
    orig_recoded_testdata2 = get_dummy_data_one_test(testname, baseline_df, timedepend_df, var_names=subscale2_names)
    # removing items from larger first instrument
    subscale1_names = collect("test1_item$(i)" for i in 1:10)
    orig_recoded_testdata_sub = get_dummy_data_one_test(testname, baseline_df, timedepend_df, var_names=subscale1_names)
else
    # read data 
    baseline_df = CSV.File(string(data_path, "baseline_df.csv"), truestrings = ["TRUE", "M"], falsestrings = ["FALSE", "F"], missingstring = ["NA"], decimal=',') |> DataFrame
    timedepend_df = CSV.File(string(data_path, "timedepend_df.csv"), truestrings = ["TRUE"], falsestrings = ["FALSE"], missingstring = ["NA"], decimal=',') |> DataFrame
    # remove "Column1"
    baseline_df = baseline_df[:,2:end]
    timedepend_df = timedepend_df[:,2:end]
    # preprocess data for a specific test 
    test = "rulm"
    testdata, sumscores, keep_timepoint_masks = get_SMArtCARE_data_one_test(test, baseline_df, timedepend_df, extended_output=true);
    # recode items 
    recoded_testdata = recode_SMArtCARE_data(testdata);
    # subscale for smaller second instrument
    subscale2_letters = ["j", "k", "l", "m", "n"]
    subscale2_names = collect("rulm_item$(letter)" for letter in subscale2_letters)
    subscale2_inds = [10, 11, 12, 13, 14]
    testdata2 = get_SMArtCARE_data_one_test(test, baseline_df, timedepend_df, var_names=subscale2_names);
    orig_recoded_testdata2 = recode_SMArtCARE_data(testdata2);
    # removing items from larger first instrument
    subscale1_letters = string.([collect('a':'i')..., collect('o':'t')...])
    subscale1_names = collect("rulm_item$(letter)" for letter in subscale1_letters)
    testdata_sub = get_SMArtCARE_data_one_test(test, baseline_df, timedepend_df, var_names=subscale1_names);
    orig_recoded_testdata_sub = recode_SMArtCARE_data(testdata_sub);
end

#------------------------------
# modification settings
#------------------------------

p_dropout = 0.5
shrink_prob_offset = 3
shift = 2
p_subgroup = 0.5

# look at distribution of sum scores
sumscores_arr = cat(sumscores..., dims=1)
sumscore_cutoff = Int(round(quantile(sumscores_arr, 0.60)))

parentdir = USE_DUMMY_DATA ? "../results/dummy/modifications/" : "../results/modifications/"
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
epochs = USE_DUMMY_DATA ? 5 : 30

# loss args
args_joint=LossArgs(
    λ_adversarialpenalty=5.0f0,
    λ_variancepenalty = penaltyweight,
    variancepenaltytype = penaltytype,
    variancepenaltyoffset = penaltyoffset, 
    λ_μpenalty=5.0f0
)

#########################################################################################
#########################################################################################
#------------------------------
# Modification 1-4+6
#------------------------------

for mod_no in [1,2,3,4,6]
    @info "Modification $(mod_no)..."

    testdata = deepcopy(orig_recoded_testdata2)
    recoded_testdata2, pathname = modify_data(
        testdata, sumscores, 
        mod_no, 
        p_dropout, shrink_prob_offset, shift, p_subgroup, sumscore_cutoff, 
        parentdir
    );

    mixeddata = SMAMixedTestData("rulm_handle_weights", "rulm", 
                                recoded_testdata2.xs, recoded_testdata.xs, 
                                recoded_testdata.xs_baseline,
                                recoded_testdata2.tvals, recoded_testdata.tvals, 
                                recoded_testdata.ids
    );

    # train jointly from scratch  

    # define model 
    modelargs1 = ModelArgs(p=size(mixeddata.xs1[1],1), 
                        q=length(mixeddata.xs_baseline[1]), 
                        dynamics=dynamics,
                        seed=1234, 
                        bottleneck=false, 
                        scale_sigmoid=2.0f0, 
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
    # paths setup
    resultspath = pathname # pathname is generated above as output of `modify_data`, and contains the number + hyperparameters of the modification 
    !isdir(resultspath) && mkdir(resultspath)
    # training hyperparams
    if mod_no == 6 # reduce variance penalty 
        penaltyweight = 2.0f0
        args_joint=LossArgs(
            λ_variancepenalty = penaltyweight,
            variancepenaltytype = penaltytype,
            variancepenaltyoffset = penaltyoffset,
            λ_adversarialpenalty=5.0f0, λ_μpenalty=5.0f0
        )
    end
    @info "Penalty weight: $(penaltyweight)"
    # model init
    Random.seed!(48)
    m1 = odevae(modelargs1);
    m2 = odevae(modelargs2);
    # prepare training
    trainingargs_joint=TrainingArgs(warmup=false, epochs=epochs, lr=lr)
    # train 
    train_mixed_model!(m1, m2, mixeddata, args_joint, trainingargs_joint, verbose=false, plotting=false)

    # create and save delta statistics and plots
    agg_abs_delta_df = make_df_from_deltas(m1, m2, mixeddata)
    # collect statistics and save to CSV 
    above_diagonal_df = collect_stats_about_deltas(agg_abs_delta_df)
    CSV.write(joinpath(resultspath, "delta_above_diagonal_stats.csv"), above_diagonal_df)
    # scatterplot
    deltaplot = create_delta_scatterplots(agg_abs_delta_df, 
        fix_limits=false,
        xlims=(0, 1.25), ylims=(0, 2.1),
        saveplot=true, 
        savepath=resultspath,
        filename="delta_scatterplot.pdf"
    )
end

#########################################################################################
#########################################################################################
#------------------------------
# No modification 
#------------------------------

# preprocess data again 
recoded_testdata2 = deepcopy(orig_recoded_testdata2)

mixeddata = SMAMixedTestData("rulm_handle_weights", "rulm", 
                            recoded_testdata2.xs, recoded_testdata.xs, 
                            recoded_testdata.xs_baseline,
                            recoded_testdata2.tvals, recoded_testdata.tvals, 
                            recoded_testdata2.ids
);

# train jointly from scratch  

# define model 
modelargs1 = ModelArgs(p=size(mixeddata.xs1[1],1), 
                    q=length(mixeddata.xs_baseline[1]), 
                    dynamics=dynamics,
                    seed=1234, 
                    bottleneck=false, 
                    scale_sigmoid=2.0f0, 
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
# paths setup
pathname = joinpath(parentdir, "no_modification")
resultspath = pathname
!isdir(resultspath) && mkdir(resultspath)
# training hyperparams
no_mod_epoch_checkpoints = [2, 5, 10, 12, 15, 18, 20, 25, 30]
penaltyweight = 5.0f0
# loss args
args_joint=LossArgs(
    λ_adversarialpenalty=5.0f0,
    λ_variancepenalty = penaltyweight,
    variancepenaltytype = penaltytype,
    variancepenaltyoffset = penaltyoffset, 
    λ_μpenalty=5.0f0
)
# model init
Random.seed!(48)
m1 = odevae(modelargs1);
m2 = odevae(modelargs2);
# train
train_modification_with_checkpoints(m1, m2, mixeddata, args_joint, no_mod_epoch_checkpoints, lr, 
    resultspath, modelargs1, modelargs2,
    save_individual_plots=false, save_delta_stats=true, fix_axis_limits=false
)

#########################################################################################
#########################################################################################
#------------------------------
# Delete items of larger instrument 
#------------------------------

# delete items of first measurement instrument: 15, 16, 17, 18, 19, 20
# corresponding to shoulder abduction and shoulder flexion: o, p, q, r, s, t

recoded_testdata_sub = deepcopy(orig_recoded_testdata_sub);
recoded_testdata2 = deepcopy(orig_recoded_testdata2);

mixeddata = SMAMixedTestData("rulm_handle_weights", "rulm_no_shoulder", 
                            recoded_testdata2.xs, recoded_testdata_sub.xs, 
                            recoded_testdata2.xs_baseline,
                            recoded_testdata2.tvals, recoded_testdata_sub.tvals, 
                            recoded_testdata2.ids
);

# train jointly from scratch  

# define model 
modelargs1 = ModelArgs(p=size(mixeddata.xs1[1],1), 
                    q=length(mixeddata.xs_baseline[1]), 
                    dynamics=dynamics,
                    seed=1234, 
                    bottleneck=false, 
                    scale_sigmoid=2.0f0, 
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
# paths setup
pathname = joinpath(parentdir, "larger_test_without_subscale")
resultspath = pathname
!isdir(resultspath) && mkdir(resultspath)
# training hyperparams
epochs = 30
penaltyweight = 5.0f0
# loss args
args_joint=LossArgs(
    λ_adversarialpenalty=5.0f0,
    λ_variancepenalty = penaltyweight,
    variancepenaltytype = penaltytype,
    variancepenaltyoffset = penaltyoffset, 
    λ_μpenalty=5.0f0
)
# init models
Random.seed!(48)
m1 = odevae(modelargs1);
m2 = odevae(modelargs2);
# prepare training
trainingargs_joint=TrainingArgs(warmup=false, epochs=epochs, lr=lr)
# train 
train_mixed_model!(m1, m2, mixeddata, args_joint, trainingargs_joint, verbose=false, plotting=false)

# create and save delta statistics and plots
agg_abs_delta_df = make_df_from_deltas(m1, m2, mixeddata)
# collect statistics and save to CSV 
above_diagonal_df = collect_stats_about_deltas(agg_abs_delta_df)
CSV.write(joinpath(resultspath, "delta_above_diagonal_stats.csv"), above_diagonal_df)
# scatterplot
deltaplot = create_delta_scatterplots(agg_abs_delta_df, 
    fix_limits=false,
    xlims=(0, 1.25), ylims=(0, 2.1),
    saveplot=true, 
    savepath=resultspath,
    filename="delta_scatterplot.pdf"
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

parentdir = USE_DUMMY_DATA ? "../results/dummy/modifications/" : "../results_RR/modifications/"

# Numbered modifications 1-6
for mod_no in [1,2,3,4,6]
    @info mod_no
    if mod_no == 1
        resultspath = joinpath(parentdir, "mod_1_p_dropout_$(p_dropout)")
    elseif mod_no == 2
        resultspath = joinpath(parentdir, "mod_2_shrink_prob_offset_$(shrink_prob_offset)")
    elseif mod_no == 3
        resultspath = joinpath(parentdir, "mod_3_shift_$(shift)")
    elseif mod_no == 4
        resultspath = joinpath(parentdir, "mod_4_shift_$(shift)_p_subgroup_$(p_subgroup)")
    elseif mod_no == 6
        resultspath = joinpath(parentdir, "mod_6_sumscore_cutoff")
    end
    modification_string = split(resultspath, "/")[end]
    mod_no == 6 && (penaltyweight = 2.0f0)
    cur_delta_df = CSV.read(joinpath(resultspath, "delta_above_diagonal_stats.csv"), DataFrame)
    cur_delta_df.epoch .= 30
    cur_delta_df.modification .= modification_string
    append!(overall_delta_stats, cur_delta_df)
end

# Larger test without subscale
modification_string = "larger_test_without_subscale"
resultspath = joinpath(parentdir, modification_string)
penaltyweight = 5.0f0
cur_delta_df = CSV.read(joinpath(resultspath, "delta_above_diagonal_stats.csv"), DataFrame)
cur_delta_df.epoch .= 30
cur_delta_df.modification .= modification_string
append!(overall_delta_stats, cur_delta_df)

# No modification
modification_string = "no_modification"
resultspath = joinpath(parentdir, modification_string)
epoch_checkpoints = [2, 5, 10, 12, 15, 18, 20, 25, 30]
penaltyweight = 5.0f0
for epoch in epoch_checkpoints
    cur_delta_df = CSV.read(joinpath(resultspath, "above_diagonal_stats_$(epoch)epochs.csv"), DataFrame)
    cur_delta_df.epoch .= epoch
    cur_delta_df.modification .= modification_string
    append!(overall_delta_stats, cur_delta_df)
end

modifications_df = filter(x -> (x.epoch == 30), overall_delta_stats)
modifications_df.perc_above_diagonal = round.(modifications_df.perc_above_diagonal, digits=3)

modifications_df = select(modifications_df, [:modification, :dimension, :perc_above_diagonal])
modifications_df[!,:perc_above_diagonal] = round.(modifications_df[!,:perc_above_diagonal], digits=3)

long_mod_df = unstack(modifications_df, :modification, :perc_above_diagonal)
long_mod_df = select!(long_mod_df, ["dimension", "no_modification", 
                                    "mod_1_p_dropout_0.5", "mod_2_shrink_prob_offset_3", 
                                    "mod_3_shift_2", "mod_4_shift_2_p_subgroup_0.5", 
                                    "mod_6_sumscore_cutoff", "larger_test_without_subscale"]
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