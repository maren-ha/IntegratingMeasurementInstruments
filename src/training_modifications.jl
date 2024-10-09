#--------------------------------------------------------------------------------
# training 
#--------------------------------------------------------------------------------

"""
    train_modification_with_checkpoints(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData,
                                         args_joint::LossArgs, epoch_checkpoints, lr::Number,
                                         configpath::String, modelargs1::ModelArgs, modelargs2::ModelArgs;
                                         save_delta_stats::Bool=true, save_delta_plots::Bool=true,
                                         save_individual_plots::Bool=true, fix_axis_limits::Bool=false,
                                         xlims=(0, 1.25), ylims=(0, 2.1))

Trains two ODE-VAE models while saving checkpoints at specified epochs. Specifically, the function provides options 
to plot and save the fitted trajectories of exemplary patients, and the delta scatterplots and summary statistics, 
which provide a way to assess the improvement in alignment between the models 
(for details, see the `make_df_from_deltas` and `collect_stats_about_deltas` functions).

# Details
1. **Epoch Loop**: 
    The function iterates over the specified epoch checkpoints, training the models for each checkpoint duration. 
    If any errors occur during training, it catches and logs them, then continues to the next checkpoint.
  
2. **Individual Patient Plots**: 
    For each checkpoint, 12 patients selected at random at the first checkpoint are used to create individual plots of the model's predictions. 
    These plots are saved as PDF files if `save_individual_plots` is set to `true`.

3. **Delta Statistics**: 
    After training at each checkpoint, the function computes and saves statistics related to the differences between
    the representations of the two models (both mean differences between latent representations, and variance of the fitted ODE trajectory). 
    It also generates scatter plots of these deltas and saves them based. 

4. **Configuration File**: 
    At the end of training (after the last checkpoint), the function writes a configuration file containing model parameters and settings.

# Arguments
- `m1::odevae`: The first ODE-VAE model to be trained.
- `m2::odevae`: The second ODE-VAE model.
- `mixeddata::SMAMixedTestData`: Dataset containing time-series and baseline data for training.
- `args_joint::LossArgs`: Loss function parameters for training the models.
- `epoch_checkpoints`: An array of epochs at which to save checkpoints and generate plots.
- `lr::Number`: Learning rate for training the models.
- `configpath::String`: Path to the configuration file where results and models will be saved.
- `modelargs1::ModelArgs`: Arguments for the first model (`m1`).
- `modelargs2::ModelArgs`: Arguments for the second model (`m2`).

# Keyword Arguments
- `save_delta_stats::Bool`: If `true`, statistics about the deltas will be saved to a CSV file. Defaults to `true`.
- `save_delta_plots::Bool`: If `true`, plots of the deltas will be saved. Defaults to `true`.
- `save_individual_plots::Bool`: If `true`, individual plots for randomly selected patients will be saved. Defaults to `true`.
- `fix_axis_limits::Bool`: If `true`, axis limits for delta plots will be fixed. Defaults to `false`.
- `xlims=(0, 1.25)`: Limits for the x-axis of delta scatter plots.
- `ylims=(0, 2.1)`: Limits for the y-axis of delta scatter plots.

# Example
```julia
train_modification_with_checkpoints(model1, model2, data, lossargs, 
                                     [10, 20, 30], 0.001, "/path/to/config", 
                                     modelargs1, modelargs2; 
                                     save_delta_stats=true, 
                                     save_delta_plots=false)
```
This will train the models `model1` and `model2` using `data`, 
saving delta statistics but not delta plots at epochs 10, 20, and 30 with a learning rate of 0.001.
"""
function train_modification_with_checkpoints(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData,
    args_joint::LossArgs, epoch_checkpoints, lr::Number, configpath::String,
    modelargs1::ModelArgs, modelargs2::ModelArgs;
    save_delta_stats::Bool=true, save_delta_plots::Bool=true, save_individual_plots::Bool=true, 
    fix_axis_limits::Bool=false, xlims=(0, 1.25), ylims=(0, 2.1))

    tmp_description = split(configpath, "/")[end]

    for epoch_ind in 1:length(epoch_checkpoints)

        epoch_name = epoch_checkpoints[epoch_ind]
        if length(epoch_checkpoints) == 1
            train_epochs = epoch_checkpoints[1]
        else
            train_epochs = [epoch_checkpoints[1], diff(epoch_checkpoints)...][epoch_ind]
        end

        # prepare training
        trainingargs_joint=TrainingArgs(warmup=false, epochs=train_epochs, lr=lr)

        # train 
        try
            train_mixed_model!(m1, m2, mixeddata, args_joint, trainingargs_joint, 
            verbose=false, plotting=false
            )
        catch e
            @info "Training stopped for setting $(tmp_description) at epoch $(epoch_name) due to error: $(e)"
            continue
        end

        # plot randomly selected individuals
        Random.seed!(789)
        plot_ids_joint = rand(mixeddata.ids, 12)
        plot_selected_ids(m1, m2, mixeddata, plot_ids_joint)

        Random.seed!(789)
        for iter in 1:5
            plot_ids_joint = rand(mixeddata.ids, 12)
            tmp = plot_selected_ids(m1, m2, mixeddata, plot_ids_joint)
            plot!(tmp, plot_title = "$(tmp_description)_$(epoch_name)epochs)")
            save_individual_plots && savefig(tmp, joinpath(configpath, "Individuals_$(iter)_$(epoch_name)epochs.pdf"))
        end

        # look at deltas
        agg_abs_delta_df = make_df_from_deltas(m1, m2, mixeddata)

        # plot
        filename = fix_axis_limits ? "deltas_scatter_$(epoch_name)epochs_fixedlims.pdf" : "deltas_scatter_$(epoch_name)epochs.pdf"
        deltaplot = create_delta_scatterplots(agg_abs_delta_df, 
            fix_limits=fix_axis_limits,
            xlims=xlims, ylims=ylims,
            saveplot=save_delta_plots, 
            savepath=configpath,
            filename=filename
        )

        # collect some statistics 
        above_diagonal_df = collect_stats_about_deltas(agg_abs_delta_df)
        # save to CSV 
        save_delta_stats && CSV.write(joinpath(configpath, "above_diagonal_stats_$(epoch_name)epochs.csv"), above_diagonal_df)

        # write config file
        if epoch_ind == length(epoch_checkpoints)
                write_model_config(configpath, modelargs1, modelargs2, args_joint, trainingargs_joint, filename="model_config.txt", writing_mode="w")
        end
    end
end

#--------------------------------------------------------------------------------
# configs 
#--------------------------------------------------------------------------------

"""
    check_and_create_config_path(resultspath::String, tmp_description::String)

Checks for the existence of a configuration path based on the provided results path and temporary description. 
If the path does not exist, it creates a new directory. The function also prints out whether a model configuration file
already exists in the directory.

# Arguments
- `resultspath::String`: The path where results and configurations are stored.
- `tmp_description::String`: A temporary description used to create a subdirectory within `resultspath`.

# Returns
- `configpath::String`: The path to the configuration directory, which is either an existing directory or a newly created one.
"""
function check_and_create_config_path(resultspath, tmp_description)

    @info tmp_description

    if isdir(joinpath(resultspath, tmp_description))
        if isfile(joinpath(resultspath, tmp_description, "model_config.txt")) 
            @info "Model config $(tmp_description) already exists"
        end
        configpath = joinpath(resultspath, tmp_description)
    else
        configpath = joinpath(resultspath, tmp_description)
        mkdir(configpath)
    end
    return configpath
end

"""
    write_model_config(
        path::String,
        modelargs1::ModelArgs, 
        modelargs2::ModelArgs,
        lossargs::LossArgs, 
        trainingargs::TrainingArgs; 
        filename::String="model_config.txt", 
        writing_mode::String="w"
    )

Writes the model configuration to a text file in the specified directory. 
The function includes details about the model setups, loss configuration, and training parameters.
It throws an error if `writing_mode` is not one of `"w"` or `"a"`.

Specifically, the function opens the specified file in the given mode (`writing_mode`). 
If the file does not exist, it will be created. The content written to the file includes:
   - A header indicating that the file contains model configuration for the joint ODE-VAE model.
   - The current date.
   - Configuration details for both models, the loss function, and the training parameters, each preceded by a section header.

# Arguments
- `path::String`: The directory path where the model configuration file will be saved.
- `modelargs1::ModelArgs`: The arguments and settings for the first model in the joint ODE-VAE setup.
- `modelargs2::ModelArgs`: The arguments and settings for the second model in the joint ODE-VAE setup.
- `lossargs::LossArgs`: The configuration settings related to the loss function used during training.
- `trainingargs::TrainingArgs`: The parameters that control the training process, such as the number of epochs and learning rates.

# Keyword Arguments
- `filename::String`: The name of the file where the configuration will be saved. Defaults to `"model_config.txt"`.
- `writing_mode::String`: Specifies the file mode for writing. 
    It can be either `"w"` for writing (overwriting any existing file) or `"a"` for appending to an existing file. Defaults to `"w"`.
"""
function write_model_config(
    path::String,
    modelargs1::ModelArgs, 
    modelargs2::ModelArgs,
    lossargs::LossArgs, 
    trainingargs::TrainingArgs; 
    filename::String="model_config.txt", 
    writing_mode::String="w"
    )

    !(writing_mode ∈ ["w", "a"]) && error("writing_mode must be either 'w' or 'a'")

    # open file and write header 
    open(joinpath(path, filename), writing_mode) do f 
        write(f, "# Model configuration for joint ODE-VAE model \n\n")
        write(f, "$(today())\n\n")
        # model 1
        write(f, "## Model setup 1 \n\n $(modelargs1)\n")
        # model 2
        write(f, "## Model setup 2 \n\n $(modelargs2)\n")
        # loss
        write(f, "## Loss setup \n\n $(lossargs)\n")
        # training
        write(f, "## Training setup \n\n $(trainingargs)\n")
    end
end