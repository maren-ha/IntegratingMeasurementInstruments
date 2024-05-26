#------------------------------
# functions to define and manipulate ODE-VAE model 
#------------------------------


#------------------------------
# define and initialize model 
#------------------------------

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

@with_kw struct LossArgs
    λ_μpenalty::Float32 = 0.0f0
    λ_ODEparamspenalty::Float32 = 0.0f0
    λ_adversarialpenalty::Float32 = 0.0f0
    λ_variancepenalty::Float32 = 0.0f0
    variancepenaltytype::Symbol = :ratio_sum # :sum_ratio, log_diff
    variancepenaltyoffset::Float32 = 1.0f0
    firstonly::Bool=false
    weighting::Bool=false
    skipt0::Bool=false
end

downscaled_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(0.01f0/sum(dims)) # smaller weights initialisation

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

function ODEparams_penalty(params)
    if length(params) > 4
        return sum(params[1:4].^2)
    else
        return sum(params.^2)
    end
end

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

latentz(μ, logσ) = μ .+ sqrt.(exp.(logσ)) .* randn(Float32,size(μ)...) # sample latent z,

kl_q_p(μ, logσ) = 0.5f0 .* sum(exp.(logσ) + μ.^2 .- 1.0f0 .- (logσ),dims=1)

#logp_x_z(m::odevae, x, z) = sum(logpdf.(Normal.(m.decodedμ(m.decoder(z)), sqrt.(exp.(m.decodedlogσ(m.decoder(z))))), x),dims=1) # get reconstruction error

function logp_x_z(m::odevae, x::AbstractVecOrMat{S}, z::AbstractVecOrMat{S}) where S <: Real 
    μ = m.decodedμ(m.decoder(z))
    logσ = m.decodedlogσ(m.decoder(z))
    res = @fastmath (-(x .- μ).^2 ./ (2.0f0 .* exp.(logσ))) .- 0.5f0 .* (log(S(2π)) .+ logσ)
    return sum(res, dims=1)
end

sqnorm(x) = sum(abs2, x)
reg(m::odevae) = sum(sqnorm, Flux.params(m.decoder,m.decodedμ,m.decodedlogσ)) # regularisation term in loss

getparams(m::odevae) = Flux.params(m.encoder, m.encodedμ, m.encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.ODEnet) # get parameters of VAE model

function getparams(m1::odevae, m2::odevae) 
    Flux.params(m1.encoder, m1.encodedμ, m1.encodedlogσ, 
                m1.decoder, m1.decodedμ, m1.decodedlogσ, 
                m2.encoder, m2.encodedμ, m2.encodedlogσ,
                m2.decoder, m2.decodedμ, m2.decodedlogσ,
                m1.ODEnet
    ) # get parameters of VAE model
end

function get_reconstruction(m::odevae, X, Y, t, args::LossArgs; sample::Bool=false)
    latentμ, latentlogσ = m.encodedμ(m.encoder(X)), m.encodedlogσ(m.encoder(X))
    params = vec(m.ODEnet(Y))
    ODEparams = m.dynamics(params)
    if args.firstonly
        smoothμ = hcat([generalsolution(tp, latentμ[:,1], ODEparams...)[1] for tp in t]...)
    else
        @assert length(t) == size(latentμ,2)
        solarray = [get_solution(startind, targetind, t, latentμ, ODEparams) for startind in 1:length(t), targetind in 1:length(t)]
        smoothμ = hcat([get_smoothμ(targetind, t, solarray, args.weighting, args.skipt0) for targetind in 1:length(t)]...)
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

function get_solution(startind, targetind, tvals, latentμ, ODEparams)
    solveatt = tvals[targetind]
    tstart = tvals[startind]
    return generalsolution(solveatt - tstart, latentμ[:,startind], ODEparams...)[1]
end

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
    loss(X, Y, t, m::odevae; args::LossArgs)

Compute the loss of the ODE-VAE model `m` on a batch of data, consisting of 
    time-dependent variables `X`, baseline variables `Y` and time point `t`. 

Details of the loss function behaviour, including additional penalties, are controlled by the 
    keyword arguments `args` of type `LossArgs`, see `?LossArgs` for details.

Returns the mean ELBO, where the ODE estimator of the underlying trajectory is used to decode the latent 
    value at the time points `t` and obtain a reconstruction according to these smooth latent dynamics 
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
    smoothμarray = hcat([get_smoothμ(targetind, [t1; t2], solarray, args.weighting, args.skipt0) for targetind in 1:lt]...)
    #solarray = [get_solution(startind, targetind, [t1;t2], hcat(latentμ1, latentμ2), ODEparams) for startind in 1:lt1+lt2, targetind in 1:lt1+lt2]

    if lt1 > 0
        if args.firstonly
            smoothμ1 = hcat([generalsolution(tp, latentμ1[:,1], ODEparams...)[1] for tp in t1]...)
        else
            #smoothμ1 = hcat([get_smoothμ(solveatt, [t1;t2], [latentμ1 latentμ2], [latentlogσ1 latentlogσ2], ODEparams, args.weighting, args.skipt0) for solveatt in t1]...)
            smoothμ1 = smoothμarray[:,t1inds]
        end
        z1 = latentz.(smoothμ1, latentlogσ1)
        ELBO1 = 1.0f0 .* logp_x_z(m1, X1, z1) .- 0.5f0 .* kl_q_p(smoothμ1, latentlogσ1)
        lossval1 = mean(-ELBO1) + 0.01f0*reg(m1)
    end
    if lt2 > 0
        if args.firstonly
            smoothμ2 = hcat([generalsolution(tp, latentμ2[:,1], ODEparams...)[1] for tp in t2]...)
        else
            #smoothμ2 = hcat([get_smoothμ(solveatt, [t1;t2], [latentμ1 latentμ2], [latentlogσ1 latentlogσ2], ODEparams, args.weighting, args.skipt0) for solveatt in t2]...
            smoothμ2 = smoothμarray[:,t2inds]
        end
        z2 = latentz.(smoothμ2, latentlogσ2)
        ELBO2 = 1.0f0 .* logp_x_z(m2, X2, z2) .- 0.5f0 .* kl_q_p(smoothμ2, latentlogσ2)
        lossval2 = mean(-ELBO2) + 0.01f0*reg(m2) # sum before 
    end
    penalties = 0.0f0
    if args.λ_μpenalty > 0.0f0
        #penalties += !isempty(t1) && args.λ_μpenalty * μ_penalty(t1, latentμ1, ODEparams)
        #penalties += !isempty(t2) && args.λ_μpenalty + μ_penalty(t2, latentμ2, ODEparams)
        penalties += !isempty(t1) && args.λ_μpenalty * sqrt.(sum(latentμ1 .- smoothμ1).^2)
        penalties += !isempty(t2) && args.λ_μpenalty * sqrt.(sum(latentμ2 .- smoothμ2).^2)
    end
    if args.λ_ODEparamspenalty > 0.0f0
        penalties += args.λ_ODEparamspenalty * ODEparams_penalty(params)
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