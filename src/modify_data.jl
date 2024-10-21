#--------------------------------------------------------------------------------
# MODIFICATIONS 
#--------------------------------------------------------------------------------

# 1) delete items at random time points 
"""
    delete_at_random!(recoded_testdata2, p_dropout, parentdir)
        
Delete items at random time points with probability `p_dropout`.

# Arguments
- `recoded_testdata2::SMATestData`: object containing the original `SMATestData` data
- `p_dropout::Float64`: probability of dropout
- `parentdir::String`: parent folder to save the modified data

# Returns
- `recoded_testdata2::SMATestData`: object containing the modified data in the form of a `SMATestData` object
- `pathname::String`: path to save the modified data
"""
function delete_at_random!(recoded_testdata2, p_dropout, parentdir)

    # delete items at random time points     
    Random.seed!(1311)
    for patient_id in recoded_testdata2.ids
        ind = findall(x -> x == patient_id, recoded_testdata2.ids)[1]
        curtvals = recoded_testdata2.tvals[ind]
        keepinds_mask = rand(length(curtvals)) .> p_dropout
        recoded_testdata2.xs[ind] = recoded_testdata2.xs[ind][:,keepinds_mask]
        recoded_testdata2.tvals[ind] = recoded_testdata2.tvals[ind][keepinds_mask]
    end

    pathname = joinpath(parentdir, "mod_1_p_dropout_$(p_dropout)")

    return recoded_testdata2, pathname
end

# 2) delete items at later time points preferentially 
"""
    delete_at_later_tps!(recoded_testdata2, shrink_prob_offset, parentdir)

Delete items at later time points preferentially.
The probability of deleting an item at time point number `t` is `t/(length(curtvals) + shrink_prob_offset)`.

# Arguments
- `recoded_testdata2::SMATestData`: object containing the original `SMATestData` data
- `shrink_prob_offset::Int`: offset for the probability of deletion
- `parentdir::String`: parent folder to save the modified data

# Returns
- `recoded_testdata2::SMATestData`: object containing the modified data in the form of a `SMATestData` object
- `pathname::String`: path to save the modified data
"""
function delete_at_later_tps!(recoded_testdata2, shrink_prob_offset, parentdir)
    
    # delete items at later time points preferentially 
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

    pathname = joinpath(parentdir, "mod_2_shrink_prob_offset_$(shrink_prob_offset)")

    return recoded_testdata2, pathname
end

# 3) apply a shift for all patients and items
"""
    shift_all_patients_items!(recoded_testdata2 shift, parentdir)

Apply a shift for all patients and items.

# Arguments
- `recoded_testdata2::SMATestData`: object containing the original `SMATestData` data
- `shift::Float64`: shift value
- `parentdir::String`: parent folder to save the modified data

# Returns
- `recoded_testdata2::SMATestData`: object containing the modified data in the form of a `SMATestData` object
- `pathname::String`: path to save the modified data
"""
function shift_all_patients_items!(recoded_testdata2, shift, parentdir)
        
    # apply a shift for all patients and items

    #shift = 2#0.5
    recoded_testdata2.xs = collect(recoded_testdata2.xs[ind] .+ shift for ind in 1:length(recoded_testdata2.xs))

    pathname = joinpath(parentdir, "mod_3_shift_$(shift)")

    return recoded_testdata2, pathname
end

# 4) apply a shift for a random subgroup of patients 
"""
    shift_random_subgroup!(recoded_testdata2, shift, p_subgroup, parentdir)

Apply a shift for a random subgroup of patients.

# Arguments
- `recoded_testdata2::SMATestData`: object containing the original `SMATestData` data
- `shift::Float64`: shift value
- `p_subgroup::Float64`: probability of a patient to be part of the shifted subgroup
- `parentdir::String`: parent folder to save the modified data

# Returns
- `recoded_testdata2::SMATestData`: object containing the modified data in the form of a `SMATestData` object
- `pathname::String`: path to save the modified data
"""
function shift_random_subgroup!(recoded_testdata2, shift, p_subgroup, parentdir)

    # apply a shift for a random subgroup of patients 

    Random.seed!(1311)
    #shift = 2#0.5
    #p_subgroup = 0.5#0.5

    for patient_id in recoded_testdata2.ids
        ind = findall(x -> x == patient_id, recoded_testdata2.ids)[1]
        if rand() < p_subgroup
            recoded_testdata2.xs[ind] = recoded_testdata2.xs[ind] .+ shift
        end
    end

    pathname = joinpath(parentdir, "mod_4_shift_$(shift)_p_subgroup_$(p_subgroup)")

    return recoded_testdata2, pathname
end

# 5) delete items at later time points if the sum score is above a certain threshold 
"""
    delete_later_above_threshold!(recoded_testdata2, sumscores, sumscore_cutoff, parentdir)

Delete items at later time points if the sum score of the other test is above a certain threshold, defined by `sumscore_cutoff`.

# Arguments
- `recoded_testdata2::SMATestData`: object containing the original `SMATestData` data
- `sumscores::Array{Array{Float64}}`: sum scores for each patient
- `sumscore_cutoff::Float64`: sum score threshold
- `parentdir::String`: parent folder to save the modified data

# Returns
- `recoded_testdata2::SMATestData`: object containing the modified data in the form of a `SMATestData` object
- `pathname::String`: path to save the modified data
"""
function delete_later_above_threshold!(recoded_testdata2, sumscores, sumscore_cutoff, parentdir)

    # delete items at later time points if the sum score is above a certain threshold 

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

    pathname = joinpath(parentdir, "mod_5_sumscore_cutoff")

    return recoded_testdata2, pathname
end

# 6) delete items at earlier time points if the sum score is above a certain threshold 
"""
    delete_earlier_above_threshold!(recoded_testdata2, sumscores, sumscore_cutoff, parentdir)

Delete items at earlier time points if the sum score of the other test is above a certain threshold, defined by `sumscore_cutoff`.

# Arguments
- `recoded_testdata2::SMATestData`: object containing the original `SMATestData` data
- `sumscores::Array{Array{Float64}}`: sum scores for each patient
- `sumscore_cutoff::Float64`: sum score threshold
- `parentdir::String`: parent folder to save the modified data

# Returns
- `recoded_testdata2::SMATestData`: object containing the modified data in the form of a `SMATestData` object
- `pathname::String`: path to save the modified data
"""
function delete_earlier_above_threshold!(recoded_testdata2, sumscores, sumscore_cutoff, parentdir)
        
    # delete items at earlier time points if the sum score is above a certain threshold 

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

    pathname = joinpath(parentdir, "mod_6_sumscore_cutoff")

    return recoded_testdata2, pathname
end

"""
    modify_data(testdata, sumscores, mod_no, 
        p_dropout, shrink_prob_offset, shift, p_subgroup, sumscore_cutoff, parentdir)

Modify the data according to the specified modification number `mod_no`. 
This function calls the corresponding function to apply the modification: 

- `mod_no = 1`: delete items at random time points (see `delete_at_random!`)
- `mod_no = 2`: delete items at later time points preferentially (see `delete_at_later_tps!`)
- `mod_no = 3`: apply a shift for all patients and items (see `shift_all_patients_items!`)
- `mod_no = 4`: apply a shift for a random subgroup of patients (see `shift_random_subgroup!`)
- `mod_no = 5`: delete items at later time points if the sum score of the other test 
            is above a certain threshold (see `delete_later_above_threshold!`)
- `mod_no = 6`: delete items at earlier time points if the sum score of the other test 
            is above a certain threshold (see `delete_earlier_above_threshold!`)

# Arguments
- `testdata::SMATestData`: object containing the original `SMATestData` data
- `sumscores::Array{Array{Float64}}`: sum scores for each patient
- `mod_no::Int`: modification number
- `p_dropout::Float64`: probability of dropout
- `shrink_prob_offset::Int`: offset for the probability of deletion
- `shift::Float64`: shift value
- `p_subgroup::Float64`: probability of a patient to be part of the shifted subgroup
- `sumscore_cutoff::Float64`: sum score threshold
- `parentdir::String`: parent folder to save the modified data

# Returns
- `recoded_testdata2::SMATestData`: object containing the modified data in the form of a `SMATestData` object
- `pathname::String`: path to save the modified data
"""
function modify_data(testdata, sumscores, mod_no, 
    p_dropout, shrink_prob_offset, shift, p_subgroup, sumscore_cutoff, 
    parentdir)

    if mod_no == 1    
        # 1) delete items at random time points 
        
        recoded_testdata2, pathname = delete_at_random!(testdata, p_dropout, parentdir
        )

    elseif mod_no == 2
        # 2) delete items at later time points preferentially 

        recoded_testdata2, pathname = delete_at_later_tps!(testdata, shrink_prob_offset, parentdir)

    elseif mod_no == 3

        # 3) apply a shift for all patients and items

        recoded_testdata2, pathname = shift_all_patients_items!(testdata, shift, parentdir)

    elseif mod_no == 4

        # 4) apply a shift for a random subgroup of patients 

        recoded_testdata2, pathname = shift_random_subgroup!(testdata, shift, p_subgroup, parentdir)

    elseif mod_no == 5

        #5) delete items at later time points if the sum score is above a certain threshold 

        recoded_testdata2, pathname = delete_later_above_threshold!(testdata, sumscores, sumscore_cutoff, parentdir)

    elseif mod_no == 6

        #6) delete items at earlier time points if the sum score is below a certain threshold 

        recoded_testdata2, pathname = delete_earlier_above_threshold!(testdata, sumscores, sumscore_cutoff, parentdir)

    else
        error("Modification number not implemented")
    end

    return recoded_testdata2, pathname
end