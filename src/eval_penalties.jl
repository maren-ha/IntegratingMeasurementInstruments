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
    # standardize each column 
    for ind in 1:n_baseline_vars
        mycol = lm_df[!,Symbol("x$ind")]
        mycol = (mycol .- mean(mycol)) ./ std(mycol)
        if all(isnan.(mycol))
            push!(nancols, Symbol("x$ind"))
        end
        lm_df[!,Symbol("x$ind")] = mycol
    end
    verbose && @info nancols
    #nancols # for zolg: 4, 11, 13, 14, 16, 19

    # convert all columns except measurement and patient_id to Float64
    for colname in names(lm_df)
        if !(colname in ["measurement", "patient_id"])
            lm_df[!,colname] = Float64.(lm_df[!,colname])
        end
    end
    # risdiplam: no x23, zolgensma: no x21
    if dataset == "all"
        linear_fm_1 = @formula(dim1 ~ 1 + t + measurement + 
            x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 
            + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24
        )
        linear_fm_2 = @formula(dim2 ~ 1 + t + measurement + 
            x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 
            + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24
        )
    elseif dataset == "risdiplam"
        linear_fm_1 = @formula(dim1 ~ 1 + t + measurement + 
            x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 
            + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x24 
        )
        linear_fm_2 = @formula(dim2 ~ 1 + t + measurement + 
            x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 
            + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 +x24
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
    lm_dim1 = fit(LinearModel, linear_fm_1, lm_df)
    lmfit_dim1 = predict(lm_dim1)
    lmprederr1 = sum((lmfit_dim1 .- lm_df[:,:dim1]).^2)

    lm_dim2 = fit(LinearModel, linear_fm_2, lm_df)
    lmfit_dim2 = predict(lm_dim2)
    lmprederr2 = sum((lmfit_dim2 .- lm_df[:,:dim2]).^2)

    prederrdf = DataFrame(
        Dimension = ["dim1", "dim2"],
        ODE = [round(ODEprederr_dim1, digits=3), round(ODEprederr_dim2, digits=3)],
        LinearModel = [round(lmprederr1, digits=3), round(lmprederr2, digits=3)]
    )

    return prederrdf
end
#

function get_reconstruction(m1::odevae, m2::odevae, X1, X2, Y, t1, t2, args::LossArgs; sample::Bool=false)

    latentμ1, latentlogσ1 = m1.encodedμ(m1.encoder(X1)), m1.encodedlogσ(m1.encoder(X1))
    #rulm latent variables
    latentμ2, latentlogσ2 = m2.encodedμ(m2.encoder(X2)), m2.encodedlogσ(m2.encoder(X2))
    #from baseline
    params = vec(m1.ODEnet(Y))
    ODEparams = m1.dynamics(params)

    lt1, lt2 = Int32(length(t1)), Int32(length(t2))
    tvals = unique([t1;t2])
    lt = length(tvals)
    # find indices of t1 and t2 in tvals
    t1inds = findall(x -> x ∈ t1, tvals)
    t2inds = findall(x -> x ∈ t2, tvals)

    solarray = [generalsolution(solveatt - [t1;t2][startind], hcat(latentμ1, latentμ2)[:,startind], ODEparams...)[1] for startind in 1:lt1+lt2, solveatt in tvals]
    smoothμarray = hcat([get_smoothμ(targetind, [t1; t2], solarray, args.weighting, args.skipt0) for targetind in 1:lt]...)

    if lt1 > 0
        if args.firstonly
            smoothμ1 = hcat([generalsolution(tp, latentμ1[:,1], ODEparams...)[1] for tp in t1]...)
        else
            smoothμ1 = smoothμarray[:,t1inds]
        end
        reconstructed_X1 = get_reconstruction_from_latentμ(m1, smoothμ1, latentlogσ1; sample=sample)
    else
        reconstructed_X1 = Matrix{Float32}(undef, size(X1,1), 0)
    end

    if lt2 > 0
        if args.firstonly
            smoothμ2 = hcat([generalsolution(tp, latentμ2[:,1], ODEparams...)[1] for tp in t2]...)
        else
            smoothμ2 = smoothμarray[:,t2inds]
        end
        reconstructed_X2 = get_reconstruction_from_latentμ(m2, smoothμ2, latentlogσ2; sample=sample)
    else
        reconstructed_X2 = Matrix{Float32}(undef, size(X2,1), 0)
    end

    return reconstructed_X1, reconstructed_X2
end
#

function get_reconstruction_from_latentμ(m::odevae, latentμ, latentlogσ; 
    sample::Bool=false)

    z = sample ? latentz.(latentμ, latentlogσ) : latentμ

    decodedμ = m.decodedμ(m.decoder(z))

    if sample 
        decodedlogσ = m.decodedlogσ(m.decoder(z))
        reconstructed_X = rand.(Normal.(decodedμ, sqrt.(exp.(decodedlogσ))))
    else
        reconstructed_X = decodedμ
    end

    return reconstructed_X

end
#

function get_reconstruction_without_ODE(m1::odevae, m2::odevae, X1, X2; 
    sample::Bool=false)

    # hfmse latent variables
    latentμ1, latentlogσ1 = m1.encodedμ(m1.encoder(X1)), m1.encodedlogσ(m1.encoder(X1))
    #rulm latent variables
    latentμ2, latentlogσ2 = m2.encodedμ(m2.encoder(X2)), m2.encodedlogσ(m2.encoder(X2))

    reconstructed_X1, reconstructed_X2 = [], []

    if size(X1,2) > 0
        reconstructed_X1 = get_reconstruction_from_latentμ(m1, latentμ1, latentlogσ1; sample=sample)
    end

    if size(X2,2) > 0
        reconstructed_X2 = get_reconstruction_from_latentμ(m2, latentμ2, latentlogσ2; sample=sample)
    end

    return reconstructed_X1, reconstructed_X2
end
#

function fit_mixed_model_on_sumscores(mixeddata::SMAMixedTestData; verbose::Bool=false)
    # fit mixed model for each patient 
    # get sum scores of each instrument 
    sumscores_origlevel_1 = collect(vec(Int.(sum(xs[1:end,:], dims=1))) for xs in mixeddata.xs1)
    sumscores_origlevel_2 = collect(vec(Int.(sum(xs[1:end,:], dims=1))) for xs in mixeddata.xs2)

    mm_df_all = DataFrame(t = [], score = [], measurement = [], patient_id = [])
    rec_scores_1 = []
    rec_scores_2 = []

    for ind in 1:length(mixeddata.ids)

        verbose && @info mixeddata.ids[ind]

        cursumscore1 = sumscores_origlevel_1[ind]
        cursumscore2 = sumscores_origlevel_2[ind]

        t1, t2 = mixeddata.tvals1[ind], mixeddata.tvals2[ind]
        lt1, lt2 = Int32(length(t1)), Int32(length(t2))

        overall_mean = mean([cursumscore1; cursumscore2])
        mean1, mean2 = mean(cursumscore1), mean(cursumscore2)

        aligned_score_1 = cursumscore1 .- mean1 .+ overall_mean
        aligned_score_2 = cursumscore2 .- mean2 .+ overall_mean

        mm_df = DataFrame(
            t = [t1;t2],
            score = [aligned_score_1; aligned_score_2],
            measurement = [fill("1", lt1); fill("2", lt2)], 
            patient_id = fill(mixeddata.ids[ind], lt1+lt2)
        )

        append!(mm_df_all, mm_df)

        if all(mm_df[:,:score] .== mm_df[1,:score])
            verbose && @info "All scores are the same for patient $(mixeddata.ids[ind])"
            mmfit = fill(mm_df[1,:score], lt1+lt2)
        else
            try
                mm_sumscore = fit(MixedModel, @formula(score ~ 1 + t + (1|measurement)), mm_df) # ranef(mm) gives random effects
                mmfit = predict(mm_sumscore)
            catch
                mmfit = fill(mm_df[1,:score], lt1+lt2)
            end
        end
        rec_score_1 = mmfit[1:lt1] .- overall_mean .+ mean1
        rec_score_2 = mmfit[lt1+1:end] .- overall_mean .+ mean2

        push!(rec_scores_1, rec_score_1)
        push!(rec_scores_2, rec_score_2)
    end
    return mm_df_all, rec_scores_1, rec_scores_2
end
# check sum scores: for RULM, they don't include "itema", so that should be left out for pure RULM
#sumscores_origlevel_from_preprocessing = collect(Vector{Int64}(sumscores[i]) for i in 1:length(sumscores))
#sumscores_origlevel = collect(vec(Int.(sum(xs[2:end,:], dims=1))) for xs in testdata.xs)
#@assert sumscores_origlevel == sumscores_origlevel_from_preprocessing

#=
tvals = unique([t1;t2])
lt = length(tvals)
# find indices of t1 and t2 in tvals
t1inds = findall(x -> x ∈ t1, tvals)
t2inds = findall(x -> x ∈ t2, tvals)

# align scores
scoremat = fill(-999, (lt, 2))
scoremat[t1inds, 1] = cursumscore1
scoremat[t2inds, 2] = cursumscore2

not1inds = setdiff(1:lt, t1inds)
not2inds = setdiff(1:lt, t2inds)

scoremat[not1inds,1] = scoremat[not1inds,2]
scoremat[not2inds,2] = scoremat[not2inds,1]

aligned_scores = fill(0.0, size(scoremat))
means = mean(scoremat, dims=1)
diffs_to_mean = fill(0.0, size(scoremat))
diffs_to_mean[:,1] = scoremat[:,1] .+ means[:,2]
diffs_to_mean[:,2] = scoremat[:,2] .- means[:,1]
diffs_to_mean = mapslices(x -> x .- vec(mean(scoremat, dims=1)), scoremat, dims=2)
aligned_scores[:,1] = scoremat[:,1] .- diffs_to_mean[:,2] # subtract diffs to mean of other instrument
aligned_scores[:,2] = scoremat[:,2] .- diffs_to_mean[:,1] # add diffs to mean of other instrument
=#