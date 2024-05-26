mutable struct SMATestData
    test::String
    xs::Vector{Matrix{Float32}}
    xs_baseline::Vector{Vector{Float32}}
    tvals::Vector{Vector{Float32}}
    ids::Vector
end

mutable struct SMAMixedTestData
    test1::String
    test2::String
    xs1::Vector{Matrix{Float32}}
    xs2::Vector{Matrix{Float32}}
    xs_baseline::Vector{Vector{Float32}}
    tvals1::Vector{Vector{Float32}}
    tvals2::Vector{Vector{Float32}}
    ids1::Vector
    ids2::Vector
    ids::Vector
end

function get_test_variables(test, colnames)
    if occursin("chop", test)
        test_vars = filter(x-> occursin("chop", x), colnames)
    elseif occursin("rulm", test)
        test_vars = filter(x-> occursin("rulm", x), colnames)
    elseif occursin("hfmse", test)
        test_vars = filter(x-> occursin("hfmse", x), colnames)
        #test_vars = filter(x -> !occursin("total_", x), test_vars)
    elseif occursin("six", test)
        test_vars = filter(x -> occursin("distance", x), colnames)
    else
        error("invalid test name")
    end
    return test_vars
end

uniqueidx(v) = unique(i -> v[i], eachindex(v))

function get_test_name(test::String)
    return "test_$test"
end

function get_score_variable_name(testname::String)
    if testname == "rulm"
        score_var_name = "rulm_score"
    end
    if testname == "hfmse"
        score_var_name = "hfmse_ex_total"
    end
    return score_var_name
end

function get_test_rows(curdf, test1, test2)
    testname1 = get_test_name(test1)
    testname2 = get_test_name(test2)
    linetest1 = []
    linetest2 = []
    lineboth = []
    for i in 1:size(curdf,1)
        if  ismissing(curdf[i,testname1]) && curdf[i,testname2]==test2
            push!(linetest2,i)
        elseif ismissing(curdf[i,testname2]) && curdf[i,testname1]==test1
            push!(linetest1,i)
        elseif curdf[i,testname1]==test1 && curdf[i,testname2]==test2
            push!(lineboth,i)
        end
    end
    curidx1=sort(vcat(linetest1,lineboth))
    curidx2=sort(vcat(linetest2,lineboth))
    return curidx1, curidx2
end

function shift_tvals!(curtvals1, curtvals2)
    if !(0 in curtvals1) & !(0 in curtvals2) 
        if isempty(curtvals1)
            curtvals2.-=minimum(curtvals2)
        elseif isempty(curtvals2)
            curtvals1.-=minimum(curtvals1)
        else
            min_of_both = min(minimum(curtvals1), minimum(curtvals2))
            curtvals1.-=min_of_both
            curtvals2.-=min_of_both
        end
    end
    return curtvals1, curtvals2
end

#test, new_baseline_df, new_timedepend_df, extended_output=true
function get_data_tests(timedepend_df, baseline_df, other_vars, baseline_vars; test1::String="hfmse", test2::String="rulm", remove_lessthan1::Bool=false)

    testname1, testname2 = get_test_name(test1), get_test_name(test2)
    notest1inds = findall(x->ismissing(x),timedepend_df[:,testname1])
    notest2inds = findall(x->ismissing(x),timedepend_df[:,testname2])
    timedepend_select_df=timedepend_df[Not(intersect(notest1inds, notest2inds)),:]
    test1_vars = get_test_variables(test1, names(timedepend_select_df))
    test2_vars = get_test_variables(test2, names(timedepend_select_df))
    select_vars = vcat(other_vars, test1_vars, test2_vars)
    timedepend_select_df=select(timedepend_select_df,select_vars)
    
    select_ids=unique(timedepend_select_df[:,:patient_id])
    #select_ids1= filter(x -> x ∈ ids, select_ids)
    #select_ids2= filter(x -> x ∉ ids, select_ids)
    #append!(select_ids1,select_ids2)
    timedepend_select_df = filter(x -> x.patient_id ∈ select_ids, timedepend_select_df)
    unique!(timedepend_select_df) #11388×77, 1215
    baseline_select_df = filter(x -> (x.patient_id ∈ select_ids), baseline_df)

    @assert sort(unique(baseline_select_df[:,:patient_id])) == sort(unique(timedepend_select_df[:,:patient_id]))
    sort!(baseline_select_df, [:patient_id])
    sort!(timedepend_select_df, [:patient_id])
    @assert unique(baseline_select_df[:,:patient_id]) == unique(timedepend_select_df[:,:patient_id])

    xs1 = []
    xs2 = []
    tvals1 = []
    tvals2 = []
    xs_baseline = []
    ids1 = []
    ids2 = []
    ids = []

    keep_timepoint_masks_1 = []
    keep_timepoint_masks_2 = []
    sumscores1 = []
    sumscores2 = []

    @showprogress for patient_id in select_ids
        # @info patient_id
        curdf = filter(x -> x.patient_id == patient_id, timedepend_select_df)

        #@time unique!(curdf)
        #if nrow(unique(curdf)) != nrow(curdf)
        #    @warn "there were duplicate rows for patient id $(patient_id), skipping these..."
        #    unique!(curdf)
        #end
        curidx1, curidx2 = get_test_rows(curdf, test1, test2)

        # other important time-dependent variables
        other_vars = ["patient_id", "months_since_1st_test"]
        #other_vars = ["patient_id", "months_since_1st_test", "feeding_tube", 
        #    "scoliosis_yn", "pain_yn", "fatigue_yn", "ventilation", "adverse_event", 
        #    "fvc_yn", "fvc_percent",
        #    "gen_impr", "mf_impr", "rf_impr"
        #]

        # test1 table
        curdf1 = select(curdf[curidx1,:],vcat(other_vars,test1_vars[test1_vars.!=testname1]))           
        curtvals1 = curdf1[:,:months_since_1st_test]
        scoreinds1 = findall(x -> occursin("score", x) || occursin("total", x), names(curdf1))

        # test2 table
        curdf2 = select(curdf[curidx2,:],vcat(other_vars,test2_vars[test2_vars.!=testname2]))           
        curtvals2 = curdf2[:,:months_since_1st_test]
        scoreinds2 = findall(x -> occursin("score", x) || occursin("total", x), names(curdf2))

        # remove patients with not more than 1 observation per test
        if length(curtvals1)<=1 && length(curtvals2)<=1 && remove_lessthan1
            continue
        end

        # remove duplicates in time points 
        @assert nrow(curdf1) == length(curtvals1)
        @assert nrow(curdf2) == length(curtvals2)
        uniqueinds_t1, uniqueinds_t2 = uniqueidx(curtvals1), uniqueidx(curtvals2)
        curtvals1, curtvals2 = curtvals1[uniqueinds_t1], curtvals2[uniqueinds_t2]
        curdf1, curdf2 = curdf1[uniqueinds_t1,:], curdf2[uniqueinds_t2,:]
        @assert (nrow(curdf1) == length(curtvals1)) && (nrow(curdf2) == length(curtvals2))

        # get sum scores 
        cursumscore1 = vec(curdf1[:,get_score_variable_name(test1)])
        cursumscore2 = vec(curdf2[:,get_score_variable_name(test2)])

        # remove score variables from curdf1 and curdf2
        curdf1 = curdf1[:,Not(scoreinds1)]
        curdf2 = curdf2[:,Not(scoreinds2)]

        #Shift tvals so that the first value starts with 0
        curtvals1, curtvals2 = shift_tvals!(curtvals1, curtvals2)
        !isempty(curtvals1) ? push!(ids1,patient_id) : push!(ids2, patient_id)

        # filter time points
        fail_counter = 0

        timedep_inds_to_keep_1 = collect(3:size(curdf1,2))
        timedep_inds_to_keep_2 = collect(3:size(curdf2,2))
        nvars1 = length(timedep_inds_to_keep_1)
        nvars2 = length(timedep_inds_to_keep_2)

        if var(cursumscore1) < 0.5 || isnan(var(cursumscore1))
            # don't record the test
            curtvals1 = Float32[]
            curxs1 = Matrix{Float32}(undef,nvars1, 0)
            fail_counter += 1
            curtimepointmask1 = nothing
            cursumscore1 = nothing
        else
            # filter time points 
            curcutoff1 = 2*iqr(diff(cursumscore1))
            curtimepointmask1 = [true; abs.(diff(cursumscore1)) .< curcutoff1...]
            cursumscore1 = cursumscore1[curtimepointmask1]
            if sum(curtimepointmask1) <= 1 
                # don't record the test
                curtvals1 = Float32[]
                curxs1 = Matrix{Float32}(undef, nvars1, 0)
                fail_counter += 1
            else
                # apply time point mask 
                curtvals1 = curtvals1[curtimepointmask1]
                curxs1 = transpose(Matrix(curdf1[curtimepointmask1,timedep_inds_to_keep_1])) # omitting first and second column because they contain ID and timestamp
                curxs1 = convert.(Float32, curxs1)
            end
        end

        if var(cursumscore2) < 0.5 || isnan(var(cursumscore2))
            # don't record the test 
            curtvals2 = Float32[]
            curxs2 = Matrix{Float32}(undef,nvars2,0)
            fail_counter += 1
            curtimepointmask2 = nothing
            cursumscore2 = nothing
        else
            # filter time points 
            curcutoff2 = 2*iqr(diff(cursumscore2))
            curtimepointmask2 = [true; abs.(diff(cursumscore2)) .< curcutoff2...]
            cursumscore2 = cursumscore2[curtimepointmask2]
            if sum(curtimepointmask2) <= 1 
                # don't record the test
                curtvals2 = Float32[]
                curxs2 = Matrix{Float32}(undef,nvars2,0)
                fail_counter += 1
            else
                # apply time point mask 
                curtvals2 = curtvals2[curtimepointmask2]
                curxs2 = transpose(Matrix(curdf2[curtimepointmask2,timedep_inds_to_keep_2])) # omitting first and second column because they contain ID and timestamp
                curxs2 = convert.(Float32, curxs2)
            end
        end

        if fail_counter > 1 
            # skip the individual
            continue 
        end

        # record time point masks 
        !isnothing(curtimepointmask1) && push!(keep_timepoint_masks_1, curtimepointmask1)
        !isnothing(curtimepointmask2) && push!(keep_timepoint_masks_2, curtimepointmask2)

        # record sum scores
        !isnothing(cursumscore1) && push!(sumscores1, cursumscore1)
        !isnothing(cursumscore2) && push!(sumscores2, cursumscore2)
        
        # baseline table 
        curdf_baseline = filter(x -> x.patient_id == patient_id, baseline_select_df[:,baseline_vars])
        curbaselinexs = vec(Matrix(curdf_baseline[:,2:end])) # again to omit patient_id

        # append vectors
        push!(ids, patient_id)
        push!(xs1, curxs1)
        push!(xs2, curxs2)
        push!(tvals1, Float32.(curtvals1))
        push!(tvals2, Float32.(curtvals2))
        push!(xs_baseline, curbaselinexs)
    end
    mixedtestdata = SMAMixedTestData(test1, test2, 
                                    convert(Vector{Matrix{Float32}},xs1), convert(Vector{Matrix{Float32}},xs2), 
                                    convert(Vector{Vector{Float32}},xs_baseline), 
                                    convert(Vector{Vector{Float32}},tvals1), convert(Vector{Vector{Float32}},tvals2), 
                                    ids1, ids2, ids)
    return mixedtestdata
end

function get_SMArtCARE_data(test::String, baseline_df, timedepend_df; var_names::Array{String}=String[], extended_output::Bool=false)

    testname="test_$test"
    
    # 1) filter timedepend df 

    # filter for patients that have the selected test conducted 
    timedepend_select_df=timedepend_df[findall(x-> !ismissing(x),timedepend_df[:,testname]),:]
    timedepend_select_df=select(timedepend_select_df,Not(testname))
    # get the variables of the items of the specific test and subset to these 
    test_vars = get_test_variables(test, names(timedepend_select_df))
    # non-item variables that are important 
    other_vars = ["patient_id", "months_since_1st_test"]
    select_vars = vcat(other_vars, test_vars)
    timedepend_select_df=select(timedepend_select_df,select_vars)

    if isempty(var_names) # if no var names are provided, take all 
        var_names = names(timedepend_select_df)
        inds_to_keep = findall(x -> !(x ∈ ["patient_id", "months_since_1st_test", "rulm_score"]), var_names)
    else
        inds_to_keep = findall(x -> x ∈ var_names, names(timedepend_select_df))
    end

    # 2) process baseline variables 

    baseline_vars = names(baseline_df)[findall(x -> !(x ∈ ["cohort", "baseline_date"]), names(baseline_df))]
    select_ids = unique(timedepend_select_df[:,:patient_id])
    baseline_select_df = filter(x -> (x.patient_id ∈ select_ids), baseline_df)
    sort!(baseline_select_df, [:patient_id])
    @assert sort(unique(baseline_select_df[:,:patient_id])) == sort(unique(timedepend_select_df[:,:patient_id]))
    @assert unique(baseline_select_df[:,:patient_id]) == unique(timedepend_select_df[:,:patient_id])
    # recode cohort variable
    for row in 1:nrow(baseline_select_df)
        if baseline_select_df[row,:cohort] == "5"
            baseline_select_df[row, :cohort2] = 9
        end
    end
    baseline_select_df = baseline_select_df[:,baseline_vars]

    # 3) collect timedependent and baseline arrays, skipping outlier time points

    # criterion: calculate IQR of temporal differences of sum score for each patient
    # for each timepoint, remove if the observed difference is outside 2*IQR

    # initialise data containers
    timedepend_xs = []
    tvals = []
    keep_timepoints_masks = []
    baseline_xs = []
    patient_ids = []
    sumscores = []

    for patient_id in select_ids
        # get timedependent variables
        curdf = filter(x -> x.patient_id == patient_id, timedepend_select_df)
        if nrow(unique(curdf)) != nrow(curdf)
            @warn "there were duplicate rows for patient id $(patient_id), skipping these..."
            unique!(curdf)
        end

        # get tvals 
        curtvals = curdf[:,:months_since_1st_test]
        if !(0 in curtvals)
            curtvals.-=minimum(curtvals)
        end
        if length(curtvals) <= 1
            continue
        end
        
        # get sum score and filter time points 
        cursumscore = vec(curdf.rulm_score)
        if var(cursumscore) < 1.0#2.0
            continue 
        end
        curcutoff = 2*iqr(diff(cursumscore))
        curtimepointmask = [true; abs.(diff(cursumscore)) .< curcutoff...]
        if sum(curtimepointmask) <= 1
            continue 
        end
        push!(keep_timepoints_masks, curtimepointmask)
        
        # apply time point mask mask
        curtvals = curtvals[curtimepointmask]
        push!(tvals, vec(curtvals))
        #curxs = transpose(Matrix(curdf[curtimepointmask,4:end])) # omitting first, second and third column because they contain ID, timestamp and sum score
        curxs = transpose(Matrix(curdf[curtimepointmask,inds_to_keep])) # omitting first, second and third column because they contain ID, timestamp and sum score
        curxs = convert.(Float32, curxs)
        push!(timedepend_xs, curxs)

        # get variance of sumscore 
        cursumscore = cursumscore[curtimepointmask]
        push!(sumscores, cursumscore)

        # get baseline variables 
        curdf_baseline = filter(x -> x.patient_id == patient_id, baseline_select_df)
        curbaselinexs = transpose(Matrix(curdf_baseline[:,2:end])) # again to omit patient_id
        curbaselinexs = vec(curbaselinexs)
        push!(baseline_xs, curbaselinexs)

        # track patient id 
        push!(patient_ids, patient_id)
    end

    # collect into testdata struct 
    testdata = SMATestData(test, 
            convert(Vector{Matrix{Float32}},timedepend_xs),
            convert(Vector{Vector{Float32}},baseline_xs), 
            convert(Vector{Vector{Float32}},tvals), 
            Int.(patient_ids)
    )
    if extended_output 
        return testdata, sumscores, keep_timepoints_masks
    else
        return testdata 
    end
end

logit(p) = log(p) - log(1-p)

function recode_SMArtCARE_data(testdata::SMATestData)
    recoding_dict = Dict(0 => 0.1, 1 => 0.5, 2 => 0.9)
    recoding_dict_itema = Dict(0 => 0.1, 1 => 0.2, 2 => 0.3, 3 => 0.5, 4 => 0.7, 5 => 0.8, 6 => 0.9)
    #plot(logit, collect(0:0.001:1))

    item_inds = collect(2:size(testdata.xs[1],1)) # without "itema", which is the first row
    recoded_xs = []
    for curxs in testdata.xs 
        if startswith(testdata.test, "rulm")
            @assert issubset(unique(curxs[2:end,:]),  [0, 1, 2])
            cur_recoded_xs = copy(curxs)
            for item_ind in item_inds
                cur_recoded_xs[item_ind,:] = [logit(recoding_dict[item_value]) for item_value in curxs[item_ind,:]]
            end
            # item a 
            cur_recoded_xs[1,:] = [logit(recoding_dict_itema[item_value]) for item_value in curxs[1,:]]
        elseif startswith(testdata.test, "hfmse")
            @assert issubset(unique(curxs),  [0, 1, 2])
            cur_recoded_xs = copy(curxs)
            for item_ind in 1:size(curxs,1)
                cur_recoded_xs[item_ind,:] = [logit(recoding_dict[item_value]) for item_value in curxs[item_ind,:]]
            end
        end
        push!(recoded_xs, cur_recoded_xs)
    end
    recoded_testdata = SMATestData(testdata.test, 
            convert(Vector{Matrix{Float32}},recoded_xs),
            testdata.xs_baseline, 
            testdata.tvals, 
            testdata.ids
    )
    return recoded_testdata
end
