function create_patients_and_visits_summary(mixeddata)

    # info about visits per patient
    n_overall = length(mixeddata.ids)
    n_test1 = length(findall(x -> size(x,2) > 0, mixeddata.xs1))
    n_test2 = length(findall(x -> size(x,2) > 0, mixeddata.xs2))

    # number of observations 
    n_obs_test1 = sum([size(x,2) for x in mixeddata.xs1])
    n_obs_test2 = sum([size(x,2) for x in mixeddata.xs2])

    min_obs_test1 = minimum([size(x,2) for x in mixeddata.xs1[findall(x -> size(x,2) > 0, mixeddata.xs1)]])
    max_obs_test1 = maximum([size(x,2) for x in mixeddata.xs1[findall(x -> size(x,2) > 0, mixeddata.xs1)]])
    median_test1 = median([size(x,2) for x in mixeddata.xs1[findall(x -> size(x,2) > 0, mixeddata.xs1)]])

    min_obs_test2 = minimum([size(x,2) for x in mixeddata.xs2[findall(x -> size(x,2) > 0, mixeddata.xs2)]])
    max_obs_test2 = maximum([size(x,2) for x in mixeddata.xs2[findall(x -> size(x,2) > 0, mixeddata.xs2)]])
    median_test2 = median([size(x,2) for x in mixeddata.xs2[findall(x -> size(x,2) > 0, mixeddata.xs2)]])

    min_total_test1 = minimum([size(mixeddata.xs1[i],2) + size(mixeddata.xs2[i],2) for i in 1:length(mixeddata.ids)])
    max_total_test1 = maximum([size(mixeddata.xs1[i],2) + size(mixeddata.xs2[i],2) for i in 1:length(mixeddata.ids)])
    median_total_test1 = median([size(mixeddata.xs1[i],2) + size(mixeddata.xs2[i],2) for i in 1:length(mixeddata.ids)])

    summary_df_patients = DataFrame(Info = [], HFMSE = [], RULM = [], overall = [])
    push!(summary_df_patients, ["Number of patients",  n_test1, n_test2, n_overall])
    push!(summary_df_patients, ["Total number of measureemnts", n_obs_test1, n_obs_test2, n_obs_test1 + n_obs_test2])
    push!(summary_df_patients, ["Minimum number of visits per patient", min_obs_test1, min_obs_test2, min_total_test1])
    push!(summary_df_patients, ["Maximum number of visits per patient", max_obs_test1, max_obs_test2, max_total_test1])
    push!(summary_df_patients, ["Median number of visits per patient", median_test1, median_test2, median_total_test1])

    # simply calculate the numbers (data probably cannot be shown anyways)
    rulm_only_per_patient = Int[]
    hfmse_only_per_patient = Int[]
    both_per_patient = Int[]
    visits_per_patient = Int[]

    # additionally calculate how many patients have only RULM / HFMSE tests or only both
    counter_only_rulm = 0
    counter_only_hfmse = 0
    counter_only_both = 0
    counter_mixed = 0

    for ind in 1:length(mixeddata.ids)

        #curtest1, curtest2 = mixeddata.xs1[ind], mixeddata.xs2[ind]
        curt1, curt2 = mixeddata.tvals1[ind], mixeddata.tvals2[ind]
        #curdf = filter(x -> x.patient_id == id, df)

        no_hfmse_only = length(findall(x -> x ∈ curt1 && x ∉ curt2, curt1))
        no_rulm_only = length(findall(x -> x ∈ curt2 && x ∉ curt1, curt2))
        no_both = length(findall(x -> x ∈ curt1 && x ∈ curt2, curt1))
        @assert no_both == length(findall(x -> x ∈ curt1 && x ∈ curt2, curt2))
        no_visits = length(curt1) + length(curt2)

        if no_both == length(curt1) && no_both == length(curt2)
            counter_only_both += 1
        elseif isempty(curt2)
            counter_only_hfmse += 1
        elseif isempty(curt1)
            counter_only_rulm += 1
        else
            counter_mixed += 1
        end

        push!(rulm_only_per_patient, no_rulm_only)
        push!(hfmse_only_per_patient, no_hfmse_only)
        push!(both_per_patient, no_both)
        push!(visits_per_patient, no_visits)
    end


    summary_df = DataFrame(Info = [], Number = [])

    # info about visits
    n_unique_timepoints = collect(length.(unique([mixeddata.tvals1[ind]..., mixeddata.tvals2[ind]...]) for ind in 1:length(mixeddata.ids)))
    n_tp_overall = sum(n_unique_timepoints)
    min_tp = minimum(n_unique_timepoints)
    max_tp = maximum(n_unique_timepoints)
    median_tp = median(n_unique_timepoints)


    # info about patients
    push!(summary_df,["Total number of patients", n_overall])

    # info about visits
    push!(summary_df, ["Total number of visits", n_tp_overall])
    push!(summary_df, ["Minimum number of visits per patient", min_tp])
    push!(summary_df, ["Maximum number of visits per patient", max_tp])
    push!(summary_df, ["Median number of visits per patient", median_tp])

    # info about measurements
    push!(summary_df,["Total number of RULM and HFMSE measurements", sum(visits_per_patient)])
    push!(summary_df,["Total number of HFMSE measurements", n_obs_test1])
    push!(summary_df,["Total number of RULM measurements", n_obs_test2])
    push!(summary_df,["Number of visits with both tests", sum(both_per_patient)])
    push!(summary_df,["Number of visits with only HFMSE", sum(hfmse_only_per_patient)])
    push!(summary_df,["Number of visits with only RULM", sum(rulm_only_per_patient)])

    push!(summary_df, ["Number of patients with HFMSE measurements", n_test1])
    push!(summary_df, ["Number of patients with RULM measurements", n_test2])

    push!(summary_df,["Number of patients with both tests at each visit", counter_only_both])
    push!(summary_df,["Number of patients with only HFMSE at each visit", counter_only_hfmse])
    push!(summary_df,["Number of patients with only RULM at each visit", counter_only_rulm])
    push!(summary_df,["Number of patients with visits with one or both tests", counter_mixed])

    # info about measurements per patient
    push!(summary_df, ["Minimum number of HFMSE visits per patient", min_obs_test1])
    push!(summary_df, ["Maximum number of HFMSE visits per patient", max_obs_test1])
    push!(summary_df, ["Median number of HFMSE visits per patient", median_test1])
    push!(summary_df, ["Minimum number of RULM visits per patient", min_obs_test2])
    push!(summary_df, ["Maximum number of RULM visits per patient", max_obs_test2])
    push!(summary_df, ["Median number of RULM visits per patient", median_test2])

    return summary_df
end

function plot_visit_patterns(plot_df, ids; seed=42)
    viz_test = plot(
        xlims = (-2, 60), xlabel="time in months",
        ylims = (0.5, 30), yticks = 0:1:30, yflip = true, ylabel="patient",
        size=(700, 1000))# , title="HFMSE and RULM measurements")

    #
    #Random.seed!(32)
    Random.seed!(seed)

    for (j, id) in enumerate(rand(ids, 30))
        curdf = filter(x -> x.patient_id == id, plot_df)
        curdf_hfmse = filter(x -> !ismissing(x.test_hfmse), curdf)
        curdf_rulm = filter(x -> !ismissing(x.test_rulm), curdf)
        if nrow(curdf_hfmse) > 0
            curdf_hfmse[!,:yval] .= j-0.15
            scatter!(viz_test, curdf_hfmse[:,:months_since_1st_test], curdf_hfmse[:,:yval], 
                    legend=false, color="#9ecae1", marker = (4, :square))
        end
        if nrow(curdf_rulm) > 0
            curdf_rulm[!,:yval] .= j+0.2
            scatter!(viz_test, curdf_rulm[:,:months_since_1st_test], curdf_rulm[:,:yval], 
                    legend=false, color="#3182bd", marker = (4, :circle))
        end
    end
    
    return viz_test

end
