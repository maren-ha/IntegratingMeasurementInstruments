function createindividualplot(m::odevae, testdata::SMATestData, args::LossArgs, patient_id; 
    axislabs::Bool=false, title::String="", showOLS::Bool=true)
    idx=findall(x -> x == patient_id, testdata.ids)

    if length(idx) > 1
        error("patient ID $patient_id not unique!")
    else
        idx = idx[1]
    end
    curxs, curxs_baseline, curtvals = testdata.xs[idx], testdata.xs_baseline[idx], testdata.tvals[idx]
    latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
    params = vec(m.ODEnet(curxs_baseline))
    ODEparams = m.dynamics(params)
    trange = Float32.(minimum(curtvals):0.1:maximum(curtvals)+1)
    if args.firstonly
        smoothμs = hcat([generalsolution(tp, latentμ[:,1], ODEparams...)[1] for tp in trange]...)
    else    
        solarray = [generalsolution(solveatt - curtvals[startind], latentμ[:,startind], ODEparams...)[1] for startind in 1:length(curtvals), solveatt in trange]
        #solarray = [get_solution(startind, targetind, curtvals, latentμ, ODEparams) for startind in 1:length(curtvals), targetind in 1:length(curtvals)]
        smoothμs = hcat([get_smoothμ(targetind, curtvals, solarray, false, false) for targetind in 1:length(trange)]...)
        #smoothμs = hcat([get_smoothμ(solveatt, curtvals, latentμ, latentlogσ, ODEparams, args.weighting, false) for solveatt in trange]...)
    end
    curplot = plot(collect(trange), smoothμs', 
        line=(3, ["#1f77b4" "#ff7f0e"]), 
        labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"]
    )
    if showOLS
        OLSfit = hcat(predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[1,:])))), predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[2,:])))))
        plot!(curtvals, OLSfit, line = (3, "#e70f4f", :dash), label ="")
    end
    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 6, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 6, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}", title="$patient_id")
    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end
    plot!(title=title)
    return curplot
end

function plot_selected_ids(m::odevae, testdata::SMATestData, args::LossArgs, selected_ids::Array; showOLS::Bool=true)
    sel_array = []
    for (ind, patient_id) in enumerate(selected_ids)
        push!(sel_array, createindividualplot(m, testdata, args, patient_id, title="$(patient_id)", showOLS=showOLS))
    end
    panelplot = plot(sel_array..., layout=(Int(length(selected_ids)/4),4), legend=false, size=(1200,round(200/3)*length(selected_ids)))
    return panelplot
end