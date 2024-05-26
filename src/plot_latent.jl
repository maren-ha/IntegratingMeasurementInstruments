#------------------------------
# individual plots
#------------------------------

function get_max_tval(tvals1::Vector{Float32}, tvals2::Vector{Float32})
    if !isempty(tvals1) && !isempty(tvals2)
        tmax = max(maximum(tvals1), maximum(tvals2))
    else
        tmax = maximum(union(tvals1, tvals2))
    end
    trange = 0.0:1.0:tmax+1
    return trange 
end

function createindividualplot(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, args::LossArgs, patient_id; 
            axislabs::Bool=false, title::String="", 
            colors_ODE = ["#1f77b4" "#ff7f0e"], colors_points=["#1f77b4" "#1f77b4"; "#ff7f0e" "#ff7f0e"],
            marker_shapes = [:c, :rect], marker_sizes = [6, 4])
            # shape of colors_points: first row μ1, second row μ2, first column test1, second column test2
    idx=findall(x -> x == patient_id, mixeddata.ids)[1]
    curxs1, curxs2, curxs_baseline, curtvals1, curtvals2 = mixeddata.xs1[idx], mixeddata.xs2[idx], mixeddata.xs_baseline[idx], mixeddata.tvals1[idx], mixeddata.tvals2[idx]
    latentμ1, latentlogσ1 = m1.encodedμ(m1.encoder(curxs1)), m1.encodedlogσ(m1.encoder(curxs1))
    latentμ2, latentlogσ2 = m2.encodedμ(m2.encoder(curxs2)), m2.encodedlogσ(m2.encoder(curxs2))
    params = vec(m1.ODEnet(curxs_baseline))
    ODEparams = m1.dynamics(params)
    trange = get_max_tval(curtvals1, curtvals2)
    # get smoothμ
    if args.firstonly # solve only at first tp like in Michelle's thesis 
        if !isempty(t1)
            smoothμ = hcat([generalsolution(tp, latentμ1[:,1], ODEparams...)[1] for tp in trange]...)
        else
            smoothμ = hcat([generalsolution(tp, latentμ2[:,1], ODEparams...)[1] for tp in trange]...)
        end
    else 
        solarray1 = [generalsolution(solveatt - curtvals1[startind], latentμ1[:,startind], ODEparams...)[1] for startind in 1:length(curtvals1), solveatt in trange]
        solarray2 = [generalsolution(solveatt - curtvals2[startind], latentμ2[:,startind], ODEparams...)[1] for startind in 1:length(curtvals2), solveatt in trange]
        smoothμ = hcat([get_smoothμ(targetind, vcat(curtvals1, curtvals2), vcat(solarray1, solarray2), false, false) for targetind in 1:length(trange)]...)
        #smoothμ = hcat([get_smoothμ(solveatt, vcat(curtvals1,curtvals2), hcat(latentμ1,latentμ2), hcat(latentlogσ1, latentlogσ2), ODEparams, args.weighting, args.skipt0) for solveatt in trange]...)
    end
    curplot = plot(collect(trange), smoothμ', 
                    line=(2, colors_ODE), 
                    labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"]
    )
    # points test 1 and 2
    Plots.scatter!(curtvals1, latentμ1[1,:], marker = (marker_shapes[1], marker_sizes[1], colors_points[1,1]), label = L"\mu_1 \mathrm{~from~encoder~test1}") 
    Plots.scatter!(curtvals1, latentμ1[2,:], marker = (marker_shapes[1], marker_sizes[1], colors_points[2,1]), label = L"\mu_2 \mathrm{~from~encoder~test1}")
    Plots.scatter!(curtvals2, latentμ2[1,:], marker = (marker_shapes[2], marker_sizes[2], colors_points[1,2]), label = L"\mu_1 \mathrm{~from~encoder~test2}") 
    Plots.scatter!(curtvals2, latentμ2[2,:], marker = (marker_shapes[2], marker_sizes[2], colors_points[2,2]), label = L"\mu_2 \mathrm{~from~encoder~test2}")
    if axislabs
        plot!(xlab="time (months)", ylab="value of latent representation")
    end
    plot!(title=title)
    return curplot
end
#

#------------------------------
# array of selected IDs
#------------------------------

function plot_selected_ids(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, args::LossArgs, selected_ids::Array; 
    colors_ODE = ["#1f77b4" "#ff7f0e"], colors_points=["#1f77b4" "#1f77b4"; "#ff7f0e" "#ff7f0e"],
    marker_shapes = [:c, :rect], marker_sizes = [6, 4]
    )
    sel_array = []
    for (ind, patient_id) in enumerate(selected_ids)
        #println(ind)
        #println(patient_id)
        push!(sel_array, createindividualplot(m1, m2, mixeddata, args, patient_id, 
            title="$(patient_id)", colors_ODE=colors_ODE, colors_points=colors_points, 
            marker_shapes=marker_shapes, marker_sizes=marker_sizes)
        )
    end
    panelplot = plot(sel_array..., layout=(Int(length(selected_ids)/4),4), legend=false, size=(1200,round(200/3)*length(selected_ids)))
    return panelplot
end

function plot_selected_ids_final(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, args::LossArgs, selected_ids::Array; 
    colors_ODE = ["#1f77b4" "#ff7f0e"], colors_points=["#1f77b4" "#1f77b4"; "#ff7f0e" "#ff7f0e"],
    marker_shapes = [:c, :rect], marker_sizes = [6, 4]
    )
    sel_array = []
    for (ind, patient_id) in enumerate(selected_ids)
        #println(ind)
        #println(patient_id)
        push!(sel_array, createindividualplot(m1, m2, mixeddata, args, patient_id, 
            title="$ind", colors_ODE=colors_ODE, colors_points=colors_points, 
            marker_shapes=marker_shapes, marker_sizes=marker_sizes)
        )
    end
    panelplot = plot(sel_array..., layout=(3,4), legend=false, size=(1200,700), margin=1mm)
    return panelplot
end