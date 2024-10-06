"""
    eval_prediction(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, n_future_tps::Int=1)

Evaluate the prediction performance of the model on the latent representations and collect 
the latent values, ODE predictions and baseline variables in a dataframe 
(for subsequent comparison with baseline methods). 

# Arguments
- `m1::odevae`: trained `odevae` model
- `m2::odevae`: trained `odevae` model
- `mixeddata`: `SMATestData` object containing the SMArtCARE data
- `n_future_tps::Int=1`: number of future time points to predict

# Returns
- `ODEprederrs1`: vector of prediction errors for the ODE model for measurement instrument 1
- `ODEprederrs2`: vector of prediction errors for the ODE model for measurement instrument 2
- `df_all`: DataFrame containing the ODE solutions, latent values, time points, patient IDs, and baseline variables 
"""
function eval_prediction(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, args::LossArgs,
    n_future_tps::Int=1; verbose::Bool=false)
    # to test: patient_id = 75598
    ODEprederrs1 = Float32[]
    ODEprederrs2 = Float32[]

    df_all = DataFrame(
        t = Float32[], 
        dim1 = Float32[], 
        dim2 = Float32[], 
        measurement = String[], 
        patient_id = String[], 
        ODE_dim1 = Float32[],
        ODE_dim2 = Float32[]
    )
    # automatically add columns for each baseline variable
    for varind in 1:length(mixeddata.xs_baseline[1])
        df_all[!, Symbol("x$varind")] = Float32[]
    end

    for patient_id in mixeddata.ids

        verbose && @info patient_id # for debugging
        
        # @info patient_id # for debugging
        
        idx=findall(x -> x == patient_id, mixeddata.ids)[1]
        curxs1, curxs2, curxs_baseline, t1, t2 = mixeddata.xs1[idx], mixeddata.xs2[idx], mixeddata.xs_baseline[idx], mixeddata.tvals1[idx], mixeddata.tvals2[idx]
        # get latent representations
        latentμ1, latentlogσ1 = m1.encodedμ(m1.encoder(curxs1)), m1.encodedlogσ(m1.encoder(curxs1))
        latentμ2, latentlogσ2 = m2.encodedμ(m2.encoder(curxs2)), m2.encodedlogσ(m2.encoder(curxs2))
        # get ODE solutions 
        params = vec(m1.ODEnet(curxs_baseline))
        ODEparams = m1.dynamics(params)
        # get unique timepoints and their indices, so they can be looked up later 
        lt1, lt2 = Int32(length(t1)), Int32(length(t2))
        tvals = unique([t1;t2])
        lt = length(tvals)
        # find indices of t1 and t2 in tvals
        t1inds = findall(x -> x ∈ t1, tvals)
        t2inds = findall(x -> x ∈ t2, tvals)
        # collect all solutions into an array 
        solarray = [generalsolution(solveatt - [t1;t2][startind], hcat(latentμ1, latentμ2)[:,startind], ODEparams...)[1] for startind in 1:lt1+lt2, solveatt in tvals]
        smoothμarray = hcat([get_smoothμ(targetind, [t1; t2], solarray, args.weighting, args.skipt0) for targetind in 1:lt]...)
    
        if !args.firstonly
            smoothμ1 = smoothμarray[:,t1inds]
            smoothμ2 = smoothμarray[:,t2inds]
        else
            error("Evaluation not implemented for `args.firstonly`=true")
        end
        
        ODEprederr1 = 1 / lt1 .* sum((latentμ1 .- smoothμ1).^2) # here, 1 and 2 refers to the measurement instruments
        ODEprederr2 = 1 / lt2 .* sum((latentμ2 .- smoothμ2).^2)

        push!(ODEprederrs1, ODEprederr1)
        push!(ODEprederrs2, ODEprederr2)

        # now, save the data for the mixed model
        curdf = DataFrame(
            t = [t1;t2],
            dim1 = [latentμ1[1,:]; latentμ2[1,:]],
            dim2 = [latentμ1[2,:]; latentμ2[2,:]],
            measurement = [fill("1", lt1); fill("2", lt2)], 
            patient_id = fill(string(patient_id), lt1+lt2),
            ODE_dim1 = [smoothμ1[1,:]; smoothμ2[1,:]],
            ODE_dim2 = [smoothμ1[2,:]; smoothμ2[2,:]]
        )

        # add baseline variables 
        baseline_vec = Float32.(mixeddata.xs_baseline[idx])'
        baseline_df= repeat(DataFrame(baseline_vec, :auto), lt1+lt2)
        curdf = hcat(curdf, baseline_df)

        append!(df_all, curdf)        
    end

    # compare to a linear model where the intercept is obtained from the baseline variables

    verbose && @info "Mean of measurement 1 ODE predictions: $(mean(ODEprederrs1[.!isnan.(ODEprederrs1)]))"
    verbose && @info "Mean of measurement 2 ODE predictions: $(mean(ODEprederrs2[.!isnan.(ODEprederrs2)]))"

    return df_all, ODEprederrs1, ODEprederrs2
end

"""
    fit_baseline_model(df_all::DataFrame; verbose::Bool=false)

Fit baseline model for comparison with the ODE model in the latent space: 
a linear model, where the intercept is learned from baseline variables, with time and measurement instrument
as further covariates

# Arguments
- `df_all::DataFrame`: DataFrame containing the ODE solutions, latent values, time points, patient IDs, and baseline variables

# Returns
- `prederrdf`: DataFrame containing the prediction errors for the ODE model and the baseline models
"""
function fit_baseline_model(df_all::DataFrame, n_baseline_vars; verbose::Bool=false, dataset::String="all")

    ODEprederr_dim1 = sum((df_all[:,:dim1] .- df_all[:,:ODE_dim1]).^2)
    ODEprederr_dim2 = sum((df_all[:,:dim2] .- df_all[:,:ODE_dim2]).^2)

    lm_df = deepcopy(df_all)
    nancols = []
    # standardize each column, columns with zero variance are later removed from the model
    for ind in 1:n_baseline_vars
        mycol = lm_df[!,Symbol("x$ind")]
        mycol = (mycol .- mean(mycol)) ./ std(mycol)
        if all(isnan.(mycol))
            push!(nancols, Symbol("x$ind"))
        end
        lm_df[!,Symbol("x$ind")] = mycol
    end
    verbose && @info nancols
    #nancols for zolg: 4, 11, 13, 14, 16, 19

    # convert all columns except measurement and patient_id to Float64
    for colname in names(lm_df)
        if !(colname in ["measurement", "patient_id"])
            lm_df[!,colname] = Float64.(lm_df[!,colname])
        end
    end
    # fit  linear models
    if dataset == "all"
        linear_fm_1 = @formula(dim1 ~ 1 + t + measurement + 
            x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 
            + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24
        )
        linear_fm_2 = @formula(dim2 ~ 1 + t + measurement + 
            x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 
            + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24
        )
    elseif dataset == "zolgensma"
        linear_fm_1 = @formula(dim1 ~ 1 + t + measurement + 
            x1 + x2 + x3 + x5 + x6 + x7 + x8 + x9 + x10 + x12 
            + x15 + x17 + x18 + x20 + x22 + x23 + x24 
        )
        linear_fm_2 = @formula(dim2 ~ 1 + t + measurement + 
        x1 + x2 + x3 + x5 + x6 + x7 + x8 + x9 + x10 + x12 
        + x15 + x17 + x18 + x20 + x22 + x23 + x24 
    )
    end
    # calculate squared errors
    lm_dim1 = fit(LinearModel, linear_fm_1, lm_df)
    lmfit_dim1 = predict(lm_dim1)
    lmprederr1 = sum((lmfit_dim1 .- lm_df[:,:dim1]).^2)

    lm_dim2 = fit(LinearModel, linear_fm_2, lm_df)
    lmfit_dim2 = predict(lm_dim2)
    lmprederr2 = sum((lmfit_dim2 .- lm_df[:,:dim2]).^2)

    # gather predictions in dataframe
    prederrdf = DataFrame(
        Dimension = ["dim1", "dim2"],
        ODE = [round(ODEprederr_dim1, digits=3), round(ODEprederr_dim2, digits=3)],
        LinearModel = [round(lmprederr1, digits=3), round(lmprederr2, digits=3)]
    )

    return prederrdf
end
