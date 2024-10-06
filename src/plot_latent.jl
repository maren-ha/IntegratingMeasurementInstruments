#------------------------------
# individual plots
#------------------------------

"""
    get_max_tval(tvals1::Vector{Float32}, tvals2::Vector{Float32})

Compute the maximum time value (`tmax`) across two time value vectors, and return a range from `0.0` to `tmax + 1` with step size `1.0`.
If either vector is empty, the function returns the maximum time value from the other vector.

# Arguments
- `tvals1::Vector{Float32}`: First vector of time values.
- `tvals2::Vector{Float32}`: Second vector of time values.

# Returns
- `trange::UnitRange{Float64}`: A range of values from `0.0` to `tmax + 1.0`, where `tmax` is the maximum time from both input vectors.
"""
function get_max_tval(tvals1::Vector{Float32}, tvals2::Vector{Float32})
    if !isempty(tvals1) && !isempty(tvals2)
        tmax = max(maximum(tvals1), maximum(tvals2))
    else
        tmax = maximum(union(tvals1, tvals2))
    end
    trange = 0.0:1.0:tmax+1
    return trange 
end

"""
    createindividualplot(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, patient_id; 
                         axislabs::Bool=false, title::String="", 
                         colors_ODE = ["#1f77b4", "#ff7f0e"], 
                         colors_points = ["#1f77b4", "#1f77b4"; "#ff7f0e", "#ff7f0e"], 
                         marker_shapes = [:c, :rect], marker_sizes = [6, 4])

Creates an individual plot for a specific patient using latent representations from two 
    ODE-VAE models (`m1` and `m2`), where both the joint latent ODE trajectory and the latent
    representations from both measurement instruments before solving the ODE are shown. 

# Arguments
- `m1::odevae`: The first model for encoding and generating latent representations.
- `m2::odevae`: The second model for encoding and generating latent representations.
- `mixeddata::SMAMixedTestData`: The dataset containing test values for multiple patients.
- `patient_id`: The ID of the patient for whom the plot is generated.

# Keyword Arguments
- `axislabs::Bool`: Whether to include axis labels. Defaults to `false`.
- `title::String`: Title for the plot. Defaults to an empty string.
- `colors_ODE::Vector{String}`: Colors for the ODE smooth trajectories. Defaults to `["#1f77b4", "#ff7f0e"]`.
- `colors_points::Matrix{String}`: Colors for the scatter points, where the rows corresponds to the 
    dimensions of the latent representation and the column to the measurement instrument.  
    Defaults to `["#1f77b4", "#1f77b4"; "#ff7f0e", "#ff7f0e"]`.
- `marker_shapes::Vector{Symbol}`: Marker shapes for measurement instrument 1 and 2 points. Defaults to `[:c, :rect]`.
- `marker_sizes::Vector{Int}`: Marker sizes for measurement instrument 1 and 2 points. Defaults to `[6, 4]`.

# Returns
- `curplot`: The generated plot displaying smooth latent representations and scatter points for the given patient.
"""
function createindividualplot(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, patient_id; 
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
    solarray1 = [generalsolution(solveatt - curtvals1[startind], latentμ1[:,startind], ODEparams...)[1] for startind in 1:length(curtvals1), solveatt in trange]
    solarray2 = [generalsolution(solveatt - curtvals2[startind], latentμ2[:,startind], ODEparams...)[1] for startind in 1:length(curtvals2), solveatt in trange]
    smoothμ = hcat([get_smoothμ(targetind, vcat(curtvals1, curtvals2), vcat(solarray1, solarray2), false, false) for targetind in 1:length(trange)]...)
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

#------------------------------
# array of selected IDs
#------------------------------
"""
    plot_selected_ids(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, selected_ids::Array;
                      colors_ODE = ["#1f77b4", "#ff7f0e"], 
                      colors_points = ["#1f77b4", "#1f77b4"; "#ff7f0e", "#ff7f0e"], 
                      marker_shapes = [:c, :rect], marker_sizes = [6, 4])

Creates a panel plot showing the latent trajectories of multiple selected patients. 
For each selected patient, the latent trajectories are plotted by calling `createindividualplot`. 
The panel layout is arranged in a 3x4 grid.

# Arguments
- `m1::odevae`: The first model for encoding and generating latent representations.
- `m2::odevae`: The second model for encoding and generating latent representations.
- `mixeddata::SMAMixedTestData`: The dataset containing test values for multiple patients.
- `selected_ids::Array`: Array of patient IDs for which to create plots.

# Keyword Arguments
- `colors_ODE::Vector{String}`: Colors for the ODE smooth trajectories. Defaults to `["#1f77b4", "#ff7f0e"]`.
- `colors_points::Matrix{String}`: Colors for the scatter points, where the rows corresponds to the 
    dimensions of the latent representation and the column to the measurement instrument.  
    Defaults to `["#1f77b4", "#1f77b4"; "#ff7f0e", "#ff7f0e"]`.
- `marker_shapes::Vector{Symbol}`: Marker shapes for test 1 and test 2 points. Defaults to `[:c, :rect]`.
- `marker_sizes::Vector{Int}`: Marker sizes for test 1 and test 2 points. Defaults to `[6, 4]`.

# Returns
- `panelplot`: A panel plot arranged in a 3x4 grid, displaying the latent trajectory plots for the selected patients.
"""
function plot_selected_ids(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData, selected_ids::Array; 
    colors_ODE = ["#1f77b4" "#ff7f0e"], colors_points=["#1f77b4" "#1f77b4"; "#ff7f0e" "#ff7f0e"],
    marker_shapes = [:c, :rect], marker_sizes = [6, 4]
    )
    sel_array = []
    for (ind, patient_id) in enumerate(selected_ids)
        #println(ind)
        #println(patient_id)
        push!(sel_array, createindividualplot(m1, m2, mixeddata, patient_id, 
            title="$ind", colors_ODE=colors_ODE, colors_points=colors_points, 
            marker_shapes=marker_shapes, marker_sizes=marker_sizes)
        )
    end
    panelplot = plot(sel_array..., layout=(3,4), legend=false, size=(1200,700), margin=1mm)
    return panelplot
end