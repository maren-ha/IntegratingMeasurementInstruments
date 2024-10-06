#--------------------------------------------------------------------------------
# MODIFICATIONS 
#--------------------------------------------------------------------------------

"""
    make_df_from_deltas(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData) 

Generate a DataFrame containing the differences (deltas) between the latent representations 
from two ODE-VAE models (`m1` and `m2`) for each patient across time points, 
and the differences between the values of the ODE solution at the first and last time point 
of the time interval for each patient. 

Specifically, for each patient, the mean absolute difference between the latent representations 
of each measurement instrument is calculated at all time points where both measurement 
instruments were observed. 

Specifically, the function computes the following:
1. Latent variable deltas between `m1` and `m2` for each patient at shared time points.
2. ODE deltas based on smooth ODE solutions for each patient.
3. A comparison between the ODE deltas and the mean absolute latent deltas for each patient and dimension.
4. A flag (`above_diagonal`) to indicate whether the ODE delta is larger than the mean absolute delta for easier analysis.

# Arguments
- `m1::odevae`: The first ODE-VAE model used for encoding time-series data.
- `m2::odevae`: The second ODE-VAE model for comparison with the first.
- `mixeddata::SMAMixedTestData`: A `SMAMixedTestData` object containing the input data, including patient IDs, baseline variables, time points, and test data for both models.

# Returns
- `DataFrame`: A DataFrame containing the following columns:
    - `patient_id`: Patient identifier.
    - `latent_dimension`: The dimension of the latent space.
    - `mean_abs_delta`: The mean absolute delta of the latent variables.
    - `ode_delta`: The delta between the first and last ODE solution for each dimension.
    - `abs_ode_delta`: The absolute value of the ODE delta.
    - `above_diagonal`: Boolean value indicating whether the ODE delta exceeds the mean absolute latent delta.
"""
function make_df_from_deltas(m1::odevae, m2::odevae, mixeddata::SMAMixedTestData)
    # look at deltas

    # for each patient one box plot of the deltas across time
    # get latent values for each test
    all_patient_deltas = []
    delta_df = DataFrame(patient_id = Int[], latent_dimension = Int[], delta = Real[], tval = Real[])
    ode_delta_df = DataFrame(patient_id = Int[], latent_dimension = Int[], ode_delta = Real[], abs_ode_delta = Real[])

    for patient_id in mixeddata.ids
        idx=findall(x -> x == patient_id, mixeddata.ids)[1]
        curxs1, curxs2, curxs_baseline, curtvals1, curtvals2 = mixeddata.xs1[idx], mixeddata.xs2[idx], mixeddata.xs_baseline[idx], mixeddata.tvals1[idx], mixeddata.tvals2[idx]
        latentμ1, latentlogσ1 = m1.encodedμ(m1.encoder(curxs1)), m1.encodedlogσ(m1.encoder(curxs1))
        latentμ2, latentlogσ2 = m2.encodedμ(m2.encoder(curxs2)), m2.encodedlogσ(m2.encoder(curxs2))
        equal_time_point_inds1 = findall(x -> x ∈ curtvals2, curtvals1)
        equal_time_point_inds2 = findall(x -> x ∈ curtvals1, curtvals2)
        patient_deltas = latentμ1[:,equal_time_point_inds1] .- latentμ2[:,equal_time_point_inds2]
        for (tp_ind, tp_val) in enumerate(equal_time_point_inds1) #equal_time_point_inds1
            #push!(delta_df, [patient_id 1 patient_deltas[1,tp_ind] round(curtvals1[tp_ind], digits=1)])
            #push!(delta_df, [patient_id 2 patient_deltas[2,tp_ind] round(curtvals1[tp_ind], digits=1)])
            push!(delta_df, [patient_id 1 patient_deltas[1,tp_ind] round(curtvals1[tp_val], digits=1)])
            push!(delta_df, [patient_id 2 patient_deltas[2,tp_ind] round(curtvals1[tp_val], digits=1)])
        end
        push!(all_patient_deltas, patient_deltas)
        # get ODE solutions to find their delta 
        params = vec(m1.ODEnet(curxs_baseline))
        ODEparams = m1.dynamics(params)
        trange = get_max_tval(curtvals1, curtvals2)
        # get smoothμ
        solarray1 = [generalsolution(solveatt - curtvals1[startind], latentμ1[:,startind], ODEparams...)[1] for startind in 1:length(curtvals1), solveatt in trange]
        solarray2 = [generalsolution(solveatt - curtvals2[startind], latentμ2[:,startind], ODEparams...)[1] for startind in 1:length(curtvals2), solveatt in trange]
        smoothμ = hcat([get_smoothμ(targetind, vcat(curtvals1, curtvals2), vcat(solarray1, solarray2), false, false) for targetind in 1:length(trange)]...)
        ode_delta_1 = smoothμ[1,end] - smoothμ[1,1]
        ode_delta_2 = smoothμ[2,end] - smoothμ[2,1]
        push!(ode_delta_df, [patient_id 1 ode_delta_1 abs(ode_delta_1)])
        push!(ode_delta_df, [patient_id 2 ode_delta_2 abs(ode_delta_2)])
    end

    # get absolute values
    abs_delta_df = copy(delta_df)
    abs_delta_df.delta = abs.(abs_delta_df.delta)

    # aggregate
    gdf_abs_delta = groupby(abs_delta_df[:, Not(:tval)], [:patient_id, :latent_dimension])
    agg_abs_delta_df = combine(gdf_abs_delta, :delta => mean => :mean_abs_delta)
    
    # add ODE dynamics 
    agg_abs_delta_df = leftjoin(agg_abs_delta_df, ode_delta_df, on = [:patient_id, :latent_dimension])
    
    # add color coding for above and beyond diagonal line 
    agg_abs_delta_df.above_diagonal = agg_abs_delta_df.abs_ode_delta .> agg_abs_delta_df.mean_abs_delta

    return agg_abs_delta_df
end

"""
    collect_stats_about_deltas(agg_abs_delta_df::DataFrame)

Collect statistics on the proportion of values above the diagonal in each latent dimension and overall, 
based on the comparison of ODE deltas and mean absolute latent deltas between the latent representations 
from both measurement instruments.

Specifically, the function computes:
1. The number and percentage of deltas above the diagonal in each latent dimension (`1`, `2`) 
    as well as the combined overall count.
2. A summary DataFrame containing the results for both dimensions and overall.


# Arguments
- `agg_abs_delta_df::DataFrame`: A DataFrame generated by `make_df_from_deltas` containing 
     - the deltas, 
     - ODE deltas, 
     - and flags for whether each delta is above the diagonal.

# Returns
- `DataFrame`: A DataFrame summarizing the number and percentage of values above the diagonal for each latent dimension and overall. The columns include:
    - `dimension`: The latent dimension (1 or 2) or "overall" for combined results.
    - `no_above_diagonal`: The number of instances where the delta was above the diagonal.
    - `no_total`: The total number of data points for each dimension.
    - `perc_above_diagonal`: The percentage of instances where the delta was above the diagonal for each dimension.
"""
function collect_stats_about_deltas(agg_abs_delta_df)
    # absolute numbers and percentages of values above/below diagonal in both dimensions and overall
    agg_abs_delta_df_1 = filter(x -> x.latent_dimension == 1, agg_abs_delta_df)
    agg_abs_delta_df_2 = filter(x -> x.latent_dimension == 2, agg_abs_delta_df)
    no_above_diagonal_1 = sum(agg_abs_delta_df_1.above_diagonal)
    no_above_diagonal_2 = sum(agg_abs_delta_df_2.above_diagonal)
    no_above_diagonal = sum(agg_abs_delta_df.above_diagonal)
    perc_above_diagonal_1 = no_above_diagonal_1/nrow(agg_abs_delta_df_1)
    perc_above_diagonal_2 = no_above_diagonal_2/nrow(agg_abs_delta_df_2)
    perc_above_diagonal = no_above_diagonal / nrow(agg_abs_delta_df)
    # collect into table and save to csv 
    above_diagonal_df = DataFrame(
        dimension = [1, 2, "overall"],
        no_above_diagonal = [no_above_diagonal_1, no_above_diagonal_2, no_above_diagonal],
        no_total = [nrow(agg_abs_delta_df_1), nrow(agg_abs_delta_df_2), nrow(agg_abs_delta_df)],
        perc_above_diagonal = [perc_above_diagonal_1, perc_above_diagonal_2, perc_above_diagonal]
    )

    return above_diagonal_df    
end

"""
    create_delta_scatterplots(agg_abs_delta_df::DataFrame; 
        markershape::Symbol=:c, markersize::Int=6, markeralpha::Float64=0.8, markerstroke=Plots.stroke(0, :white), 
        markercolors=["#636363", "#c7c7c7"], linecolor::String="#be0028", linewidth::Number=3,
        plotsize::Tuple=(1100, 500), fix_limits::Bool=false, xlims=(0, 1.25), ylims=(0, 2.1),
        saveplot::Bool=false, savepath::String="", filename::String="deltas_scatter.pdf")

Create scatterplots comparing the mean absolute delta of latent dimensions with ODE-based deltas for both latent dimensions, 
color-coded based on whether the deltas are above or below the diagonal.

# Arguments
- `agg_abs_delta_df::DataFrame`: A DataFrame containing the deltas, ODE deltas, and a flag indicating 
    whether each point is above the diagonal (created as output of the function `make_df_from_deltas`).

# Keyword Arguments
- `markershape::Symbol`: Shape of the markers in the scatterplot. Default is `:c`.
- `markersize::Int`: Size of the markers. Default is `6`.
- `markeralpha::Float64`: Transparency level of the markers. Default is `0.8`.
- `markerstroke`: Stroke styling for the markers. Default is `Plots.stroke(0, :white)`.
- `markercolors::Vector{String}`: Colors for markers above and below the diagonal. Default is `["#636363", "#c7c7c7"]`.
- `linecolor::String`: Color of the diagonal line. Default is `"#be0028"`.
- `linewidth::Number`: Width of the diagonal line. Default is `3`.
- `plotsize::Tuple`: Size of the final plot. Default is `(1100, 500)`.
- `fix_limits::Bool`: Whether to fix the x and y limits to specified values. Default is `false`.
- `xlims::Tuple`: Limits for the x-axis. Default is `(0, 1.25)`.
- `ylims::Tuple`: Limits for the y-axis. Default is `(0, 2.1)`.
- `saveplot::Bool`: Whether to save the plot to a file. Default is `false`.
- `savepath::String`: Path to save the plot file. Default is `""`.
- `filename::String`: Filename for saving the plot. Default is `"deltas_scatter.pdf"`.

# Returns
- `finalplot`: A plot with two scatterplots for latent dimensions 1 and 2, placed side by side.
"""
function create_delta_scatterplots(agg_abs_delta_df; 
    markershape::Symbol=:c, markersize::Int=6, markeralpha::Float64=0.8, markerstroke=Plots.stroke(0, :white), 
    markercolors = ["#636363", "#c7c7c7"], linecolor::String = "#be0028", linewidth::Number = 3,
    plotsize::Tuple = (1100, 500), fix_limits::Bool=false, xlims = (0, 1.25), ylims = (0, 2.1),
    saveplot::Bool=false, savepath::String="", filename::String = "deltas_scatter.pdf"
    )
    # first dimension 
    plot_dim1 = scatter(
        filter(x -> (x.latent_dimension .== 1) && (x.above_diagonal .== 1), agg_abs_delta_df)[:,:mean_abs_delta],
        filter(x -> (x.latent_dimension .== 1) && (x.above_diagonal .== 1), agg_abs_delta_df)[:,:abs_ode_delta],
        m = (markershape, markersize, markeralpha, markerstroke, markercolors[1]), #7f7f7f
        xlabel = L"mean absolute $\Delta$ between $(\mu^R)_1$ and $(\mu^S)_1$", 
        ylabel = L"absolute difference $\mid \tilde{\mu}(t_{\mathrm{max}})_1 - \tilde{\mu}(t_0)_1\mid$", 
        title = "Latent dimension 1", 
        legend = false
    )
    scatter!(
        filter(x -> (x.latent_dimension .== 1) && (x.above_diagonal .== 0), agg_abs_delta_df)[:,:mean_abs_delta],
        filter(x -> (x.latent_dimension .== 1) && (x.above_diagonal .== 0), agg_abs_delta_df)[:,:abs_ode_delta],
        m = (markershape, markersize, markeralpha, markerstroke, markercolors[2]), #bdbdbd
    )
    Plots.abline!(1, 0, linewidth = linewidth, color = linecolor)
    fix_limits && plot!(xlims = xlims, ylims = ylims)

    plot_dim2 = scatter(
        filter(x -> (x.latent_dimension .== 2) && (x.above_diagonal .== 1), agg_abs_delta_df)[:,:mean_abs_delta],
        filter(x -> (x.latent_dimension .== 2) && (x.above_diagonal .== 1), agg_abs_delta_df)[:,:abs_ode_delta],
        m = (markershape, markersize, markeralpha, markerstroke, markercolors[1]), #7f7f7f
        xlabel = L"mean absolute $\Delta$ between $(\mu^R)_2$ and $(\mu^S)_2$", 
        ylabel = L"absolute difference $\mid \tilde{\mu}(t_{\mathrm{max}})_2 - \tilde{\mu}(t_0)_2\mid$", 
        title = "Latent dimension 2", 
        legend = false
    )
    scatter!(
        filter(x -> (x.latent_dimension .== 2) && (x.above_diagonal .== 0), agg_abs_delta_df)[:,:mean_abs_delta],
        filter(x -> (x.latent_dimension .== 2) && (x.above_diagonal .== 0), agg_abs_delta_df)[:,:abs_ode_delta],
        m = (markershape, markersize, markeralpha, markerstroke, markercolors[2]), #bdbdbd
    )
    Plots.abline!(1, 0, linewidth = linewidth, color = linecolor)
    fix_limits && plot!(xlims = xlims, ylims = ylims)

    finalplot = plot(plot_dim1, plot_dim2, layout = (1,2), size = plotsize, margin=5mm)

    saveplot && savefig(finalplot, joinpath(savepath,filename))

    return finalplot
end
