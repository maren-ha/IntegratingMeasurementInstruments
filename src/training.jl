"""
    TrainingArgs(; warmup::Bool=true, epochs_warmup::Int=0, epochs::Int=20, 
                  lr_warmup::AbstractFloat=0.001, lr::AbstractFloat=0.0005)

A structure that holds training arguments for model training. 
It uses keyword arguments to configure various training parameters 
such as an optional warmup phase, learning rates, and the number of epochs.

# Keyword Arguments
- `warmup::Bool`: Indicates whether a warmup phase should be used before the main training. Defaults to `true`.
- `epochs_warmup::Int`: The number of epochs to use for warmup, if warmup is enabled. Defaults to `0`.
- `epochs::Int`: The total number of training epochs after the warmup phase. Defaults to `20`.
- `lr_warmup::AbstractFloat`: The learning rate during the warmup phase. Defaults to `0.001`.
- `lr::AbstractFloat`: The learning rate during the main training phase. Defaults to `0.0005`.

# Example
```julia
args = TrainingArgs(warmup=true, epochs_warmup=5, epochs=50, lr_warmup=0.002, lr=0.001)
```
"""
@with_kw mutable struct TrainingArgs
    warmup::Bool=true
    epochs_warmup::Int=0
    epochs::Int=20
    lr_warmup::AbstractFloat=0.001
    lr::AbstractFloat=0.0005
end

"""
    train_mixed_model!(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, 
                       lossargs::LossArgs, trainingargs::TrainingArgs; 
                       verbose::Bool=true, plotting::Bool=false)

Joint training of two `odevae` models, such that in the two latent spaces are aligned, 
by solving a joint ODE and using the solution of the joint ODE as deoder input for each model. 
Optionally, controlled by the `lossargs`, penalty terms are used to further improve alignment. 

The function includes options for warmup, verbose logging, and periodic plotting during training.

# Details
1. **Warmup Phase (Optional)**:
   If warmup is enabled in `trainingargs`, a warmup phase runs for a specified number of epochs (`epochs_warmup`) using only the parameters of `m2` and a warmup learning rate (`lr_warmup`). The model parameters are updated using the ADAM optimizer.

2. **Main Training Phase**:
   The primary training phase runs for the number of epochs specified in `trainingargs.epochs`. Both models, `m1` and `m2`, are trained jointly with the main learning rate (`lr`). The optimizer used for both models is ADAM.

3. **Plotting**:
   If `plotting=true`, the function generates a panel of scatter plots during training for 12 randomly selected patients. This provides visual feedback of how the models are fitting the data.

4. **Random Seed**:
   The function uses random seeding for the warmup and training phases to ensure reproducibility.

# Arguments
- `m1::odevae`: The first ODE-VAE model.
- `m2::odevae`: The second ODE-VAE model.
- `mixeddata::SMAMixedTestData`: The dataset containing time-series and baseline data for training the models.
- `lossargs::LossArgs`: Loss function parameters to be used during training.
- `trainingargs::TrainingArgs`: Training arguments (e.g., learning rates, epochs) encapsulated in a `TrainingArgs` struct.

# Keyword Arguments
- `verbose::Bool`: If `true`, the function will print the mean loss at each epoch. Defaults to `true`.
- `plotting::Bool`: If `true`, scatter plots for a random selection of patients will be generated and displayed during training. Defaults to `false`.
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
        plotting && display(plot_selected_ids(m1, m2, mixeddata, plot_ids_joint))
    end
end

# FOR DEBUGGING 
# ind = ...
# X1, X2, Y, t1, t2 = mixeddata.xs1[ind], mixeddata.xs2[ind], mixeddata.xs_baseline[ind], mixeddata.tvals1[ind], mixeddata.tvals2[ind]