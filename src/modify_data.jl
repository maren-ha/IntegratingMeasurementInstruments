#--------------------------------------------------------------------------------
# MODIFICATIONS 
#--------------------------------------------------------------------------------

# 1) delete items at random time points 
function delete_at_random(test, baseline_df, timedepend_df, 
    subscale2_names, p_dropout, pathname)

    # delete items at random time points 

    testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
    xs2, xs_baseline2, tvals2, ids2 = testdata2.xs, testdata2.xs_baseline, testdata2.tvals, testdata2.ids;
    recoded_testdata2 = recode_SMArtCARE_data(testdata2);
    
    Random.seed!(1311)

    #p_dropout = 0.5#0.3

    for patient_id in recoded_testdata2.ids
        ind = findall(x -> x == patient_id, recoded_testdata2.ids)[1]
        curtvals = recoded_testdata2.tvals[ind]
        keepinds_mask = rand(length(curtvals)) .> p_dropout
        recoded_testdata2.xs[ind] = recoded_testdata2.xs[ind][:,keepinds_mask]
        recoded_testdata2.tvals[ind] = recoded_testdata2.tvals[ind][keepinds_mask]
    end

    recoded_testdata2.test="rulm_handle_weights"

    pathname = joinpath(pathname, "mod_1_p_dropout_$(p_dropout)")

    return recoded_testdata2, pathname
end

# 2) delete items at later time points preferentially 
function delete_at_later_tps(test, baseline_df, timedepend_df, 
    subscale2_names, shrink_prob_offset, pathname)
    
    # delete items at later time points preferentially 

    testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
    xs2, xs_baseline2, tvals2, ids2 = testdata2.xs, testdata2.xs_baseline, testdata2.tvals, testdata2.ids;
    recoded_testdata2 = recode_SMArtCARE_data(testdata2);

    recoded_testdata2 = recode_SMArtCARE_data(testdata2);
    Random.seed!(1311)
    #shrink_prob_offset = 3#2

    for patient_id in recoded_testdata2.ids
        ind = findall(x -> x == patient_id, recoded_testdata2.ids)[1]
        curtvals = recoded_testdata2.tvals[ind]
        ps = collect(i/(length(curtvals) + shrink_prob_offset) for i in 1:length(curtvals))
        keepinds_mask = rand(length(curtvals)) .> ps
        @assert length(keepinds_mask) == size(recoded_testdata2.xs[ind],2)
        recoded_testdata2.xs[ind] = recoded_testdata2.xs[ind][:,keepinds_mask]
        recoded_testdata2.tvals[ind] = recoded_testdata2.tvals[ind][keepinds_mask]
    end

    recoded_testdata2.test="rulm_handle_weights"

    pathname = joinpath(pathname, "mod_2_shrink_prob_offset_$(shrink_prob_offset)")

    return recoded_testdata2, pathname
end

# 3) apply a shift for all patients and items
function shift_all_patients_items(test, baseline_df, timedepend_df, 
    subscale2_names, shift, pathname)
        
    # apply a shift for all patients and items

    testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
    xs2, xs_baseline2, tvals2, ids2 = testdata2.xs, testdata2.xs_baseline, testdata2.tvals, testdata2.ids;
    recoded_testdata2 = recode_SMArtCARE_data(testdata2);

    #shift = 2#0.5
    recoded_testdata2.xs = collect(recoded_testdata2.xs[ind] .+ shift for ind in 1:length(recoded_testdata2.xs))
    recoded_testdata2.test="rulm_handle_weights"

    pathname = joinpath(pathname, "mod_3_shift_$(shift)")

    return recoded_testdata2, pathname
end

# 4) apply a shift for a random subgroup of patients 
function shift_random_subgrop(test, baseline_df, timedepend_df, 
    subscale2_names, shift, p_subgroup, pathname)

    # apply a shift for a random subgroup of patients 

    testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
    xs2, xs_baseline2, tvals2, ids2 = testdata2.xs, testdata2.xs_baseline, testdata2.tvals, testdata2.ids;
    recoded_testdata2 = recode_SMArtCARE_data(testdata2);

    Random.seed!(1311)
    #shift = 2#0.5
    #p_subgroup = 0.5#0.5

    for patient_id in recoded_testdata2.ids
        ind = findall(x -> x == patient_id, recoded_testdata2.ids)[1]
        if rand() < p_subgroup
            recoded_testdata2.xs[ind] = recoded_testdata2.xs[ind] .+ shift
        end
    end

    recoded_testdata2.test="rulm_handle_weights"

    pathname = joinpath(pathname, "mod_4_shift_$(shift)_p_subgroup_$(p_subgroup)")

    return recoded_testdata2, pathname
end

# 5) delete items at later time points if the sum score is above a certain threshold 
function delete_later_above_threshold(test, baseline_df, timedepend_df, 
    subscale2_names, sumscores, sumscore_cutoff, pathname)

    # delete items at later time points if the sum score is above a certain threshold 

    testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
    xs2, xs_baseline2, tvals2, ids2 = testdata2.xs, testdata2.xs_baseline, testdata2.tvals, testdata2.ids;
    recoded_testdata2 = recode_SMArtCARE_data(testdata2);

    for patient_id in recoded_testdata2.ids
        ind = findall(x -> x == patient_id, recoded_testdata2.ids)[1]
        cursumscore = sumscores[ind]
        counter_above = 0
        @assert length(cursumscore) == size(recoded_testdata2.xs[ind],2)
        for tp_ind in 1:length(cursumscore)
            if cursumscore[tp_ind] > sumscore_cutoff
                counter_above += 1
                if counter_above > 1
                    recoded_testdata2.xs[ind] = recoded_testdata2.xs[ind][:,1:tp_ind] # tp_ind-1? 
                    recoded_testdata2.tvals[ind] = recoded_testdata2.tvals[ind][1:tp_ind]
                    break
                end
            end
            #elseif counter_above = 1
            #    counter_above = 0
            #end
        end
    end

    recoded_testdata2.test="rulm_handle_weights"

    pathname = joinpath(pathname, "mod_5_sumscore_cutoff_$(sumscore_cutoff)")

    return recoded_testdata2, pathname
end

# 6) delete items at earlier time points if the sum score is above a certain threshold 
function delete_earlier_above_threshold(test, baseline_df, timedepend_df, 
    subscale2_names, sumscores, sumscore_cutoff, pathname)
        
    # delete items at earlier time points if the sum score is above a certain threshold 

    testdata2 = get_SMArtCARE_data(test, baseline_df, timedepend_df, var_names=subscale2_names);
    xs2, xs_baseline2, tvals2, ids2 = testdata2.xs, testdata2.xs_baseline, testdata2.tvals, testdata2.ids;
    recoded_testdata2 = recode_SMArtCARE_data(testdata2);

    for patient_id in recoded_testdata2.ids
        ind = findall(x -> x == patient_id, recoded_testdata2.ids)[1]
        cursumscore = sumscores[ind]
        counter_above = 0
        @assert length(cursumscore) == size(recoded_testdata2.xs[ind],2)
        for tp_ind in 1:length(cursumscore)
            if cursumscore[tp_ind] > sumscore_cutoff
                counter_above += 1
                if counter_above > 0#1
                    recoded_testdata2.xs[ind] = recoded_testdata2.xs[ind][:,tp_ind:end] # tp_ind-1? 
                    recoded_testdata2.tvals[ind] = recoded_testdata2.tvals[ind][tp_ind:end]
                    break
                end
            end
            #elseif counter_above = 1
            #    counter_above = 0
            #end
        end
        if counter_above == 0
            recoded_testdata2.xs[ind] = Matrix{Float32}(undef,size(recoded_testdata2.xs[ind],1), 0)
            recoded_testdata2.tvals[ind] = Float32[]
        end
    end

    recoded_testdata2.test="rulm_handle_weights"

    pathname = joinpath(pathname, "mod_6_sumscore_cutoff_$(sumscore_cutoff)")

    return recoded_testdata2, pathname
end

function modify_data(test, 
    baseline_df, timedepend_df, subscale2_names, sumscores, mod_no, 
    p_dropout, shrink_prob_offset, shift, p_subgroup, sumscore_cutoff, 
    pathname)

    if mod_no == 1    
        # 1) delete items at random time points 
        
        recoded_testdata2, pathname = delete_at_random(test, baseline_df, timedepend_df, 
            subscale2_names, p_dropout, pathname
        )

    elseif mod_no == 2
        # 2) delete items at later time points preferentially 

        recoded_testdata2, pathname = delete_at_later_tps(test, baseline_df, timedepend_df, 
            subscale2_names, shrink_prob_offset, pathname
        )

    elseif mod_no == 3

        # 3) apply a shift for all patients and items

        recoded_testdata2, pathname = shift_all_patients_items(test, baseline_df, timedepend_df, 
            subscale2_names, shift, pathname
        )

    elseif mod_no == 4

        # 4) apply a shift for a random subgroup of patients 

        recoded_testdata2, pathname = shift_random_subgrop(test, baseline_df, timedepend_df, 
            subscale2_names, shift, p_subgroup, pathname
        )

    elseif mod_no == 5

        #5) delete items at later time points if the sum score is above a certain threshold 

        recoded_testdata2, pathname = delete_later_above_threshold(test, baseline_df, timedepend_df, 
            subscale2_names, sumscores, sumscore_cutoff, pathname
        )

    elseif mod_no == 6

        #6) delete items at earlier time points if the sum score is below a certain threshold 

        recoded_testdata2, pathname = delete_earlier_above_threshold(test, baseline_df, timedepend_df, 
            subscale2_names, sumscores, sumscore_cutoff, pathname
        )

    else
        error("Modification number not implemented")
    end

    return recoded_testdata2, pathname
end