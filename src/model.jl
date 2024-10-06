#------------------------------
# functions to define and manipulate ODE-VAE model 
#------------------------------


#------------------------------
# define and initialize model
#------------------------------

"""
    odevae

Struct for an ODE-VAE model, with the following fields:
- `p`: number of VAE input dimensions, i.e., number of time-dependent variables
- `q`: number of input dimensions for the baseline neural net, i.e., number of baseline variables
- `zdim`: number of latent dimensions
- `ODEnet`: neural net to map baseline variables to individual-specific ODE parameters 
    (number of ODE parameters depends on the ODE system specified by the `dynamics` function)
- `encoder`: neural net to map input data to latent space
- `encodedμ`: neural net layer parameterizing the mean of the latent space
- `encodedlogσ`: neural net layer parameterizing the log variance of the latent space
- `decoder`: neural net to map latent variable to reconstructed input data
- `decodedμ`: neural net layer parameterizing the mean of the reconstructed input data
- `decodedlogσ`: neural net layer parameterizing the log variance of the reconstructed input data
- `dynamics`: one of `params_fullinhomogeneous`, `params_offdiagonalinhomogeneous`, 
    `params_diagonalinhomogeneous`, `params_driftonly`, `params_fullhomogeneous`, 
    `params_offdiagonalhomogeneous`, `params_diagonalhomogeneous`: function to map a parameter vector
    (=the output of the `ODEnet`) to the system matrix and constant vector of the ODE system
"""
mutable struct odevae
    p::Int
    q::Int
    zdim::Int
    ODEnet
    encoder
    encodedμ 
    encodedlogσ 
    decoder
    decodedμ 
    decodedlogσ 
    dynamics::Function # either ODEprob or params for analytical solution function 
end

"""
    ModelArgs

Struct to store model arguments, can be constructed with keyword arguments to set the following fields:
- `p`: number of VAE input dimensions, i.e., number of time-dependent variables
- `q`: number of input dimensions for the baseline neural net, i.e., number of baseline variables
- `zdim`: number of latent dimensions
- `dynamics`: one of `params_fullinhomogeneous`, `params_offdiagonalinhomogeneous`, 
    `params_diagonalinhomogeneous`, `params_driftonly`, `params_fullhomogeneous`, 
    `params_offdiagonalhomogeneous`, `params_diagonalhomogeneous`: function to map a parameter vector
    (=the output of the `ODEnet`) to the system matrix and constant vector of the ODE system
- `seed`: random seed for reproducibility
- `bottleneck`: whether to use a bottleneck layer in the `ODEnet` 
    to reduce the number of effective parameters for higher-dimensional systems
- `init_scaled`: whether to initialize the `ODEnet` with scaled weights
- `scale_sigmoid`: scaling factor for the sigmoid function used to shift the ODE parameters 
    to a sensible range, acting as a prior
- `add_diagonal`: whether to add a diagonal transformation to output of the `ODEnet` to add
    flexibility after the sigmoid transformation
"""
@with_kw struct ModelArgs
    p::Int
    q::Int
    zdim::Int=2
    dynamics::Function
    seed::Int=1234
    bottleneck::Bool=false
    init_scaled::Bool=false
    scale_sigmoid::Real=1
    add_diagonal::Bool=true
end

"""
    LossArgs

Struct to store loss arguments, can be constructed with keyword arguments to set the following fields:
- `λ_μpenalty`: weight for the penalty that encourages consistency of the mean before and after solving the ODEs
- `λ_adversarialpenalty`: weight for the adversarial penalty that encourages alignment of the two tests
- `λ_variancepenalty`: weight for the penalty on the variance of the ODE estimator
- `variancepenaltytype`: one of `:ratio_sum`, `:sum_ratio`, `:log_diff`: 
    type of penalty on the variance of the ODE estimator
- `variancepenaltyoffset`: offset used in the penalty on the variance of the latent space
"""
@with_kw struct LossArgs
    λ_μpenalty::Float32 = 0.0f0
    λ_adversarialpenalty::Float32 = 0.0f0
    λ_variancepenalty::Float32 = 0.0f0
    variancepenaltytype::Symbol = :ratio_sum # :sum_ratio, log_diff
    variancepenaltyoffset::Float32 = 1.0f0
end

"""
    downscaled_glorot_uniform(dims...) -> Array{Float32}

Generates an array of random values initialized using a downscaled version of the Glorot uniform initialization method.

# Arguments
- `dims...`: A variable number of integer arguments specifying the dimensions of the output array.

# Returns
- `Array{Float32}`: An array of the specified dimensions, filled with random values drawn from a uniform distribution scaled down to have a smaller variance. The values are centered around zero.
"""
downscaled_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(0.01f0/sum(dims)) # smaller weights initialisation

"""
    get_nODEparams(dynamics::Function) -> Int

Get the number of ODE parameters for a given ODE system specified by the function `dynamics`.

# Arguments
- `dynamics::Function`: Function specifying the ODE system, one of `params_fullinhomogeneous`, 
    `params_offdiagonalinhomogeneous`, `params_diagonalinhomogeneous`, `params_driftonly`, 
    `params_fullhomogeneous`, `params_offdiagonalhomogeneous`, `params_diagonalhomogeneous`.

# Returns
- `Int`: Number of ODE parameters for the specified ODE system.
"""
function get_nODEparams(dynamics::Function)
    if dynamics ∈ [params_driftonly, params_diagonalhomogeneous]
        nODEparams = 2
    elseif dynamics ∈[params_fullhomogeneous, params_offdiagonalinhomogeneous, params_diagonalinhomogeneous]
        nODEparams = 4 
    elseif dynamics == params_fullinhomogeneous
        nODEparams = 6 
    else
        error("unsupported model dynamics")
    end
    return nODEparams
end

# initialise model
"""
    odevae(modelargs::ModelArgs)

Function to initialize the ODE-VAE model according to the arguments passed in `modelargs`.

Returns an `odevae` model.
"""
function odevae(modelargs::ModelArgs)
    nODEparams = get_nODEparams(modelargs.dynamics)
    myinit = modelargs.init_scaled ? downscaled_glorot_uniform : Flux.glorot_uniform 
    shift(arg) = (sigmoid(arg).-0.5f0)/modelargs.scale_sigmoid
    # seed
    Random.seed!(modelargs.seed)
    # parameter network
    if !modelargs.bottleneck
        ODEnet = [Dense(modelargs.q, modelargs.q, tanh, init=myinit),
                        Dense(modelargs.q, nODEparams, arg ->(shift(arg)), init=myinit)
        ]
    else
        ODEnet = [Dense(modelargs.q, nODEparams, tanh, init=myinit),
                    Dense(nODEparams, 2, tanh, init=myinit),
                    Dense(2, nODEparams, arg ->(shift(arg)), init=myinit)
        ]
    end
    if modelargs.add_diagonal
        ODEnet = Chain(ODEnet..., Flux.Diagonal(nODEparams))
    else
        ODEnet = Chain(ODEnet...)
    end
    #   VAE encoder
    Dz, Dh = modelargs.zdim, modelargs.p
    encoder, encodedμ, encodedlogσ = Dense(modelargs.p, Dh, arg ->(tanh.(arg) .+ 1)), Dense(Dh, Dz), Chain(Dense(Dh, Dz, arg -> -Flux.relu(arg)), Flux.Diagonal(Dz))
    # VAE decoder
    decoder, decodedμ, decodedlogσ = Dense(Dz, Dh, tanh), Dense(Dh, modelargs.p), Dense(Dh, modelargs.p)

    model = odevae(modelargs.p, modelargs.q, modelargs.zdim, ODEnet, encoder, encodedμ, encodedlogσ, decoder, decodedμ, decodedlogσ, modelargs.dynamics)
    return model
end

#------------------------------
# define penalties
#------------------------------

"""
    μ_penalty(datatvals, latentμ, ODEparams) -> Float32

Compute the penalty that encourages consistency of the mean before and after solving the ODEs.

# Arguments
- `datatvals`: Vector of time points at which the data is observed
- `latentμ`: Matrix of latent variables, with each column corresponding to a time point in `datatvals`
- `ODEparams`: Tuple of the system matrix and constant vector of the ODE system

# Returns
- `Float32`: Penalty value
"""
function μ_penalty(datatvals, latentμ, ODEparams)
    penalty = 0.0f0
    for i in 1:size(latentμ,2) 
        for (tind, solveatt) in enumerate(datatvals) # make pred with current x0 for every tval
            pred, varfactor = generalsolution(solveatt - datatvals[i], latentμ[:,i], ODEparams...)
            if solveatt != datatvals[i]
                penalty += sqrt(sum((pred .- latentμ[:,tind]).^2)) # squared difference between prediction and actual datapoint 
            end
        end
    end
    return penalty/(length(datatvals)^2)
end

#------------------------------
# define model functions 
#------------------------------

"""
    latentz(μ, logσ) -> Array{Float32}

Sample latent variable `z` from the mean `μ` and log variance `logσ` using the reparameterization trick from VAE training.

# Arguments
- `μ`: Mean of the latent variable
- `logσ`: Log variance of the latent variable

# Returns
- `Array{Float32}`: Sampled latent variable `z`
"""
latentz(μ, logσ) = μ .+ sqrt.(exp.(logσ)) .* randn(Float32,size(μ)...) # sample latent z,

"""
    kl_q_p(μ, logσ) -> Array{Float32}

Compute the KL divergence between the approximate posterior `q` and a standard Normal prior `p` in the latent space.

# Arguments
- `μ`: Mean of the latent variable
- `logσ`: Log variance of the latent variable

# Returns
- `Array{Float32}`: KL divergence between `q` and `p`
"""
kl_q_p(μ, logσ) = 0.5f0 .* sum(exp.(logσ) + μ.^2 .- 1.0f0 .- (logσ),dims=1)

#logp_x_z(m::odevae, x, z) = sum(logpdf.(Normal.(m.decodedμ(m.decoder(z)), sqrt.(exp.(m.decodedlogσ(m.decoder(z))))), x),dims=1) # get reconstruction error
"""
    logp_x_z(m::odevae, x, z) -> Array{Float32}

Compute the log likelihood of the data `x` given the latent variable `z` and the ODE-VAE model `m`.

# Arguments
- `m::odevae`: ODE-VAE model
- `x`: Data
- `z`: Latent variable

# Returns
- `Array{Float32}`: Log likelihood of the data given the latent variable
"""
function logp_x_z(m::odevae, x::AbstractVecOrMat{S}, z::AbstractVecOrMat{S}) where S <: Real 
    μ = m.decodedμ(m.decoder(z))
    logσ = m.decodedlogσ(m.decoder(z))
    res = @fastmath (-(x .- μ).^2 ./ (2.0f0 .* exp.(logσ))) .- 0.5f0 .* (log(S(2π)) .+ logσ)
    return sum(res, dims=1)
end

"""
    sqnorm(x) -> Float32

Compute the squared norm of the input vector `x`.
"""
sqnorm(x) = sum(abs2, x)

"""
    reg(m::odevae) -> Float32

Compute the squared norm of the decoder parameters as regularisation term in the loss function for the ODE-VAE model `m`.

# Arguments
- `m::odevae`: ODE-VAE model

# Returns
- `Float32`: Regularisation term in loss
"""
reg(m::odevae) = sum(sqnorm, Flux.params(m.decoder,m.decodedμ,m.decodedlogσ)) # regularisation term in loss

"""
    getparams(m::odevae) -> Flux.Params

Collect the parameters of the ODE-VAE model `m` into a `Flux.Params` object.

# Arguments
- `m::odevae`: ODE-VAE model

# Returns
- `Flux.Params`: Parameters of the ODE-VAE model
"""
getparams(m::odevae) = Flux.params(m.encoder, m.encodedμ, m.encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.ODEnet) # get parameters of VAE model

"""
    getparams(m1::odevae, m2::odevae) -> Flux.Params

Collect the parameters of the ODE-VAE models `m1` and `m2` into a `Flux.Params` object.

# Arguments
- `m1::odevae`: first ODE-VAE model
- `m2::odevae`: second ODE-VAE model

# Returns
- `Flux.Params`: Parameters of the ODE-VAE models
"""
function getparams(m1::odevae, m2::odevae) 
    Flux.params(m1.encoder, m1.encodedμ, m1.encodedlogσ, 
                m1.decoder, m1.decodedμ, m1.decodedlogσ, 
                m2.encoder, m2.encodedμ, m2.encodedlogσ,
                m2.decoder, m2.decodedμ, m2.decodedlogσ,
                m1.ODEnet
    ) # get parameters of VAE model
end

"""
    get_reconstruction(m::odevae, X, Y, t, args::LossArgs; sample::Bool=false) -> Array{Float32}

Compute the decoder reconstruction of the input data `X` and baseline variables `Y`, at observed time points `t`,
using the ODE-VAE model `m`. 

# Arguments
- `m::odevae`: ODE-VAE model
- `X`: time-dependent data
- `Y`: Baseline variables
- `t`: Time points

# Keyword Arguments
- `sample::Bool=false`: Whether to sample the latent variable or use the mean for decoding

# Returns
- `Array{Float32}`: Decoder reconstruction of the input data
"""
function get_reconstruction(m::odevae, X, Y, t; sample::Bool=false)
    latentμ, latentlogσ = m.encodedμ(m.encoder(X)), m.encodedlogσ(m.encoder(X))
    params = vec(m.ODEnet(Y))
    ODEparams = m.dynamics(params)
    if args.firstonly
        smoothμ = hcat([generalsolution(tp, latentμ[:,1], ODEparams...)[1] for tp in t]...)
    else
        @assert length(t) == size(latentμ,2)
        solarray = [get_solution(startind, targetind, t, latentμ, ODEparams) for startind in 1:length(t), targetind in 1:length(t)]
        smoothμ = hcat([get_smoothμ(targetind, t, solarray, true, true) for targetind in 1:length(t)]...)
    end
    if sample
        z = latentz.(smoothμ, latentlogσ)
    else
        z = smoothμ
    end
    decodedμ = m.decodedμ(m.decoder(z))
    reconstructed_X = decodedμ
    if sample
        decodedlogσ = m.decodedlogσ(m.decoder(z))
        reconstructed_X = rand.(Normal.(decodedμ, sqrt.(exp.(decodedlogσ))))
    end
    return reconstructed_X
end

"""
    get_solution(startind, targetind, tvals, latentμ, ODEparams) -> Array{Float32}

Compute the solution of a linear ODE system for the latent variable `latentμ` at time points `tvals`,
starting from `tvals[startind]` and ending at `tvals[targetind]` using the ODE parameters `ODEparams`.
This function calls the explicit analytical solution of the ODE system.

# Arguments
- `startind`: Index of the starting time point
- `targetind`: Index of the ending time point
- `tvals`: Vector of time points
- `latentμ`: Matrix of latent variables
- `ODEparams`: Tuple of the system matrix and constant vector of the ODE system

# Returns
- `Array{Float32}`: Solution of the ODE system
"""
function get_solution(startind, targetind, tvals, latentμ, ODEparams)
    solveatt = tvals[targetind]
    tstart = tvals[startind]
    return generalsolution(solveatt - tstart, latentμ[:,startind], ODEparams...)[1]
end


"""
    get_smoothμ(targetind, tvals, solarray, weighting::Bool, skipt0::Bool) -> Array{Float32}

Compute the smoothed latent variable according to the ODE solution at time point `tvals[targetind]` using the solutions in `solarray`.

# Arguments
- `targetind`: Index of the target time point
- `tvals`: Vector of observed time points
- `solarray`: Array of solutions of the ODE system
- `weighting::Bool`: Whether to weight the predictions using inverse variance weights
- `skipt0::Bool`: Whether to skip the first time point 

# Returns
- `Array{Float32}`: Smoothed latent variable
"""
function get_smoothμ(targetind::Int, tvals::Vector{Float32}, solarray, weighting::Bool, skipt0::Bool)
    weightedpred = sum_weights = zero(solarray[1,1])
    for startind in 1:length(tvals)
        if skipt0 && (startind == targetind) && (length(tvals) > 1)
            continue
        end    
        pred = solarray[startind,targetind]
        #@info pred 
        if weighting
            # for every starting point between solveatt and datatvals[i], make a prediction for solveatt, 
            # take the empirical variance of the solutions
            var_range = startind < targetind ? (startind:targetind) : (targetind:startind)
            weight = 1.0f0 ./ var(solarray[var_range, targetind])
        else
            weight = one.(pred)
        end
        weightedpred += pred.*weight
        sum_weights += weight
    end
    return weightedpred ./ sum_weights
end

#------------------------------
# loss function
#------------------------------

"""
    loss(X1, X2, Y, t1, t2, m1::odevae, m2::odevae; args::LossArgs) -> Float32

Compute the joint loss of the ODE-VAE models `m1` and `m2` on a batch of data, consisting of 
    time-dependent variables of the first measurement instrument `X1` and of the second 
    measurement instrument `X2`, together the the patient's baseline variables `Y` and 
    observation time points `t1` of the first measurement instrument, and `t2` of the second 
    measurement instrument.  

Details of the loss function behaviour, including additional penalties, are controlled by the 
    keyword arguments `args` of type `LossArgs`, see `?LossArgs` for details.

Returns the sum of the mean ELBOs of `m1` and `m2`, where the ODE estimator of the underlying 
    joint trajectory based on the latent representations of `X1` and `X2` is used as input for 
    the decoder reconstructions of `m1` and `m2`, respectively, to obtain reconstructions of 
    `X1` at time points `t1` and of `X2` at `t2`, according to these smooth latent dynamics 
    as specified by the ODE system.
"""
function loss(X1, X2, Y, t1, t2, m1::odevae, m2::odevae; args::LossArgs)
    #hfmse latent variables
    latentμ1, latentlogσ1 = m1.encodedμ(m1.encoder(X1)), m1.encodedlogσ(m1.encoder(X1))
    #rulm latent variables
    latentμ2, latentlogσ2 = m2.encodedμ(m2.encoder(X2)), m2.encodedlogσ(m2.encoder(X2))
    #from baseline
    params = vec(m1.ODEnet(Y))
    ODEparams = m1.dynamics(params)
    lossval1,lossval2 = 0.0f0, 0.0f0

    lt1, lt2 = Int32(length(t1)), Int32(length(t2))
    tvals = unique([t1;t2])
    lt = length(tvals)
    # find indices of t1 and t2 in tvals
    t1inds = findall(x -> x ∈ t1, tvals)
    t2inds = findall(x -> x ∈ t2, tvals)

    solarray = [generalsolution(solveatt - [t1;t2][startind], hcat(latentμ1, latentμ2)[:,startind], ODEparams...)[1] for startind in 1:lt1+lt2, solveatt in tvals]
    smoothμarray = hcat([get_smoothμ(targetind, [t1; t2], solarray, true, true) for targetind in 1:lt]...)
    #solarray = [get_solution(startind, targetind, [t1;t2], hcat(latentμ1, latentμ2), ODEparams) for startind in 1:lt1+lt2, targetind in 1:lt1+lt2]

    if lt1 > 0
        smoothμ1 = smoothμarray[:,t1inds]
        z1 = latentz.(smoothμ1, latentlogσ1)
        ELBO1 = 1.0f0 .* logp_x_z(m1, X1, z1) .- 0.5f0 .* kl_q_p(smoothμ1, latentlogσ1)
        lossval1 = mean(-ELBO1) + 0.01f0*reg(m1)
    end
    if lt2 > 0
        smoothμ2 = smoothμarray[:,t2inds]
        z2 = latentz.(smoothμ2, latentlogσ2)
        ELBO2 = 1.0f0 .* logp_x_z(m2, X2, z2) .- 0.5f0 .* kl_q_p(smoothμ2, latentlogσ2)
        lossval2 = mean(-ELBO2) + 0.01f0*reg(m2) # sum before 
    end
    penalties = 0.0f0
    if args.λ_μpenalty > 0.0f0
        penalties += !isempty(t1) && args.λ_μpenalty * sqrt.(sum(latentμ1 .- smoothμ1).^2)
        penalties += !isempty(t2) && args.λ_μpenalty * sqrt.(sum(latentμ2 .- smoothμ2).^2)
    end
    if args.λ_adversarialpenalty > 0.0f0
        diff1 = isempty(t1) ? 0.0f0 : Float32(1.0f0/lt1).*sum(smoothμ1 .- latentμ1,dims=2)
        diff2 = isempty(t2) ? 0.0f0 : Float32(1.0f0/lt2).*sum(smoothμ2 .- latentμ2,dims=2)
        penalties += args.λ_adversarialpenalty * ((lt1+lt2)/5.0f0) * sum((diff1 .- diff2).^2)
    end
    if args.λ_variancepenalty > 0.0f0
        offset = args.variancepenaltyoffset
        if args.variancepenaltytype::Symbol == :ratio_sum
            var_ratio1 = lt1>1 ? (sum(mean(var(solarray[1:lt1,targetind]) for targetind in t1inds)) .+ offset) / (sum(var(latentμ1, dims=2)) .+ offset) : 0.0f0
            var_ratio2 = lt2>1 ? (sum(mean(var(solarray[lt1+1:lt1+lt2,targetind]) for targetind in t2inds)) .+ offset) / (sum(var(latentμ2, dims=2)) .+ offset) : 0.0f0
        elseif args.variancepenaltytype::Symbol == :sum_ratio
            var_ratio1 = lt1>1 ? sum((mean(var(solarray[1:lt1,targetind]) for targetind in t1inds) .+ offset) ./ (var(latentμ1, dims=2) .+ offset)) : 0.0f0
            var_ratio2 = lt2>1 ? sum((mean(var(solarray[lt1+1:lt1+lt2,targetind]) for targetind in t2inds) .+ offset) ./ (var(latentμ2, dims=2) .+ offset)) : 0.0f0
        elseif args.variancepenaltytype::Symbol == :log_diff
            #var_ratio = !isempty(t1) && sum(mean(log.(var(solarray[:,targetind]) .+ offset) for targetind in 1:lt1+lt2) .- log.(var(hcat(latentμ1, latentμ2), dims=2) .+ offset))
            var_ratio1 = lt1>1 ? sum(mean(log.(var(solarray[1:lt1,targetind]) .+ offset) for targetind in t1inds) .- log.(var(latentμ1, dims=2) .+ offset)) : 0.0f0
            var_ratio2 = lt2>1 ? sum(mean(log.(var(solarray[lt1+1:lt1+lt2,targetind]) .+ offset) for targetind in t2inds) .- log.(var(latentμ2, dims=2) .+ offset)) : 0.0f0
        end
        penalties += args.λ_variancepenalty * (var_ratio1 + var_ratio2)
    end
    lossval = lossval1 + lossval2 + penalties
    return lossval 
end