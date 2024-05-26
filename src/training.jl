@with_kw mutable struct TrainingArgs
    warmup::Bool=true
    epochs_warmup::Int=0
    epochs::Int=20
    lr_warmup::AbstractFloat=0.001
    lr::AbstractFloat=0.0005
end

"""
    train_model!(m::odevae, 
        xs, xs_baseline, tvals, 
        lr, epochs, args::LossArgs; 
        selected_ids=nothing, 
        verbose::Bool=true, 
        plotting::Bool=true
        )

Train the ODE-VAE model `m` on a dataset of time-dependent variables `xs`, 
    baseline variables `xs_baseline` and time points `tvals`. The structure of these 
    is assumed to be as in the `SMATestData` and `simdata` structs. 

# Arguments
- `m`: the ODE-VAE model to train
- `xs`: a vector of matrices of time-dependent variables for each patient
- `xs_baseline`: a vector of vectors of baseline variables for each patient
- `tvals`: a vector of vectors of time points for each patient
- `lr`: the learning rate of the ADAM optimizer
- `epochs`: the number of epochs to train for
- `args`: arguments controlling the loss function behaviour, see `?LossArgs` for details
- `selected_ids`: the IDs of the patients to plot during training to monitor progress,
    if `nothing` (default) then 12 random IDs are selected
- `verbose`: whether to print the epoch and loss value during training
- `plotting`: whether to visualize the learnt latent trajectories of selected patients 
    (those with the `selected_ids`)

# Returns 
- `m`: the trained ODE-VAE model
"""
function train_mixed_model!(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, 
                            lossargs::LossArgs, trainingargs::TrainingArgs; 
                            verbose::Bool=true, plotting::Bool=false)
    #ps1 = getparams(m1)
    @info "initializing training..."
    ps2 = getparams(m2)
    ps_joint = getparams(m1, m2)

    opt_warmup = ADAM(trainingargs.lr_warmup)
    opt_joint = ADAM(trainingargs.lr)

    trainingdata_joint = zip(mixeddata.xs1, mixeddata.xs2, mixeddata.xs_baseline, mixeddata.tvals1, mixeddata.tvals2);

    if plotting 
        Random.seed!(789)
        plot_ids_joint = rand(mixeddata.ids, 12)
    end

    if verbose 
        evalcb() = @show(mean(loss(data..., m1, m2, args=lossargs) for data in trainingdata_joint))
    end

    # optionally: start warmup 
    if trainingargs.warmup 
        @info "starting warmup for $(trainingargs.epochs_warmup) epochs... "
        state = copy(Random.default_rng());
        for epoch in 1:trainingargs.epochs_warmup
            @info epoch
            copy!(Random.default_rng(), state);
                for (X1, X2, Y, t1, t2) in trainingdata_joint
                    grads = Flux.gradient(ps2) do 
                        loss(X1, X2, Y, t1, t2, m1, m2, args=lossargs)
                    end
                    Flux.Optimise.update!(opt_warmup, ps2, grads)
                end
            state = copy(Random.default_rng());
            verbose && evalcb()
            plotting && display(plot_selected_ids(m1, m2, mixeddata, lossargs, plot_ids_joint))
        end
        @info "warmup done!"
    end
    # start training 
    @info "starting training for $(trainingargs.epochs) epochs"
    state = copy(Random.default_rng());
    for epoch in 1:trainingargs.epochs
        @info epoch
        #counter = 0
        copy!(Random.default_rng(), state);
            for (X1, X2, Y, t1, t2) in trainingdata_joint
                #counter += 1
                #@info counter
                grads = Flux.gradient(ps_joint) do 
                #grads = Flux.gradient(ps2) do 
                        loss(X1, X2, Y, t1, t2, m1, m2, args=lossargs)
                end
                Flux.Optimise.update!(opt_joint, ps_joint, grads)
                #Flux.Optimise.update!(opt_joint, ps2, grads)
            end
        state = copy(Random.default_rng());
        verbose && evalcb()
        plotting && display(plot_selected_ids(m1, m2, mixeddata, lossargs, plot_ids_joint))
    end
end

# FOR DEBUGGING 
# ind = ...
# X1, X2, Y, t1, t2 = mixeddata.xs1[ind], mixeddata.xs2[ind], mixeddata.xs_baseline[ind], mixeddata.tvals1[ind], mixeddata.tvals2[ind]