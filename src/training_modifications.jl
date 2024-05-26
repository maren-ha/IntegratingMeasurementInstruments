#--------------------------------------------------------------------------------
# training 
#--------------------------------------------------------------------------------

function do_training_checkpoints(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData,
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
        plot_selected_ids(m1, m2, mixeddata, args_joint, plot_ids_joint)

        Random.seed!(789)
        for iter in 1:5
            plot_ids_joint = rand(mixeddata.ids, 12)
            tmp = plot_selected_ids(m1, m2, mixeddata, args_joint, plot_ids_joint)
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