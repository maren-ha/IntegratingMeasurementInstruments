#------------------------------
# Setup
#------------------------------

cd(@__DIR__)

using Pkg;
Pkg.activate("../.")
Pkg.status()

using CSV
using Dates
using DataFrames
using Distributions
using Random
using LaTeXStrings
using LinearAlgebra
using Plots 
using ProgressMeter
using StatsBase
using VegaLite

gr()

sourcedir = "../src/"
include(joinpath(sourcedir, "load_data.jl"))
include(joinpath(sourcedir, "explore_data.jl"))

results_path_datainfos = "../results/dataset_infos/"
!isdir(results_path_datainfos) && mkdir(results_path_datainfos)

#------------------------------
#------------------------------
# NUSINERSEN
#------------------------------
#------------------------------

results_path_nusi = joinpath(results_path_datainfos, "nusinersen/")
!isdir(results_path_nusi) && mkdir(results_path_nusi)

#------------------------------
# load and preprocess data
#------------------------------

data_path = joinpath("../dataset/") 

baseline_df = CSV.File(string(data_path, "baseline_df.csv"), truestrings = ["TRUE", "M"], falsestrings = ["FALSE", "F"], missingstring = ["NA"], decimal=',') |> DataFrame
timedepend_df = CSV.File(string(data_path, "timedepend_df.csv"), truestrings = ["TRUE"], falsestrings = ["FALSE"], missingstring = ["NA"], decimal=',') |> DataFrame

# remove "Column1"
baseline_df = baseline_df[:,Not("Column1")]
timedepend_df = timedepend_df[:,Not("Column1")]

# copy out some preprocessing from get_SMArtCARE_data_one_test function (in load_data.jl) to get some statistics 
test1="hfmse"
test2="rulm"

other_vars = ["patient_id", "months_since_1st_test"]
baseline_vars = names(baseline_df)[findall(x -> !(x ∈ ["cohort", "baseline_date"]), names(baseline_df))]

mixeddata = get_SMArtCARE_data_two_tests(timedepend_df, baseline_df, other_vars, baseline_vars; test1=test1, test2=test2, remove_lessthan1=true);
final_ids = Int.(mixeddata.ids)

# extended preprocessing to get timedepend_select_df with correct ids 
testname1, testname2 = get_test_name(test1), get_test_name(test2)
notest1inds = findall(x->ismissing(x),timedepend_df[:,testname1])
notest2inds = findall(x->ismissing(x),timedepend_df[:,testname2])
timedepend_select_df=timedepend_df[Not(intersect(notest1inds, notest2inds)),:]
test1_vars = get_test_variables(test1, names(timedepend_select_df))
test2_vars = get_test_variables(test2, names(timedepend_select_df))
select_vars = vcat(other_vars, test1_vars, test2_vars)
timedepend_select_df=select(timedepend_select_df,select_vars)
# filter to ids after preprocessing
timedepend_select_df = filter(x -> x.patient_id ∈ final_ids, timedepend_select_df) 

#------------------------------
# basic statistics: create and save patients summary 
#------------------------------

n_obs_hfmse = length(findall(x -> !ismissing(x), timedepend_select_df[:,testname1])) # 4040
n_obs_rulm = length(findall(x -> !ismissing(x), timedepend_select_df[:,testname2])) # 4222

# both
both_inds = findall(x -> !ismissing(x.test_hfmse) && !ismissing(x.test_rulm), eachrow(timedepend_select_df)) # 3603
both_df = timedepend_select_df[both_inds,[:patient_id, :test_hfmse, :test_rulm, :months_since_1st_test]]
n_obs_both = length(unique(both_df[:,:patient_id])) # 492
# hfmse only
hfmse_only_inds = findall(x -> !ismissing(x.test_hfmse) && ismissing(x.test_rulm), eachrow(timedepend_select_df)) # 437
hfmse_only_df = timedepend_select_df[hfmse_only_inds,[:patient_id, :test_hfmse, :test_rulm, :months_since_1st_test]]
n_obs_only_h = length(unique(hfmse_only_df[:,:patient_id])) # 162
# rulm only
rulm_only_inds = findall(x -> ismissing(x.test_hfmse) && !ismissing(x.test_rulm), eachrow(timedepend_select_df)) # 619
rulm_only_df = timedepend_select_df[rulm_only_inds,[:patient_id, :test_hfmse, :test_rulm, :months_since_1st_test]]
n_obs_only_r = length(unique(rulm_only_df[:,:patient_id])) # 140

# dataset statistics
@info "test1:" mixeddata.test1
@info "test2:" mixeddata.test2

summary_df_nusi = create_patients_and_visits_summary(mixeddata)
CSV.write(joinpath(results_path_nusi, "summary_patients_and_visits.csv"), summary_df_nusi)

# typical age distribution 
baseline_df = filter(x -> x.patient_id ∈ final_ids, baseline_df)
age_at_first_treatment_nusi = baseline_df[:,:age_at_first_treatment]
presymptomatic_nusi = sum(baseline_df[:,:onset_presymptomatic])/nrow(baseline_df)

#------------------------------
# visualize typical patterns
#------------------------------

# viz 
plot_df_nusi = timedepend_select_df[:,[:patient_id, :test_hfmse, :test_rulm, :months_since_1st_test]]
seed = 17
viz_test_nusi = plot_visit_patterns(plot_df_nusi, final_ids; seed=seed)
savefig(viz_test_nusi, joinpath(results_path_nusi, "visits_per_patient_sample_$(seed).pdf"))

#------------------------------
#------------------------------
# NEW DATASET (for Risdiplam and Zolgensma)
#------------------------------
#------------------------------

data_path_total = joinpath("../dataset/")

# load total data 
baseline_total_df = CSV.File(string(data_path_total, "baseline_df_total.csv"), truestrings = ["TRUE", "M"], falsestrings = ["FALSE", "F"], missingstring = ["NA"], decimal=',') |> DataFrame
timedepend_total_df = CSV.File(string(data_path_total, "timedepend_df_total.csv"), truestrings = ["TRUE"], falsestrings = ["FALSE"], missingstring = ["NA"], decimal=',') |> DataFrame

# remove Column1
baseline_total_df = baseline_total_df[:,2:end]
timedepend_total_df = timedepend_total_df[:,2:end]

other_vars = ["patient_id", "months_since_1st_test"]
baseline_total_vars = names(baseline_total_df)[findall(x -> !(x ∈ ["cohort", "baseline_date", "birth_d", "drug_sma"]), names(baseline_total_df))]

# preprocess baseline_total_df 
# replace missings in mutation by false
baseline_total_df[ismissing.(baseline_total_df[:,:mutation_exon]), :mutation_exon] .= false
baseline_total_df[ismissing.(baseline_total_df[:,:mutation_compound]), :mutation_compound] .= false
baseline_total_df[ismissing.(baseline_total_df[:,:mutation_other]), :mutation_other] .= false
baseline_total_df[ismissing.(baseline_total_df[:,:presym_diag]), :presym_diag] .= false
# calculate age based on birth date and baseline data, assuming that the first treatment was at baseline
rows_with_missing_age = findall(x -> ismissing(x), baseline_total_df[:,:age_at_first_treatment])
imputed_ages_days = baseline_total_df[rows_with_missing_age,:baseline_date] .- baseline_total_df[rows_with_missing_age,:birth_d]
imputed_ages_months = Int.(round.(Dates.value.(imputed_ages_days) ./ 30.5))
baseline_total_df[rows_with_missing_age,:age_at_first_treatment] = imputed_ages_months

#------------------------------
#------------------------------
# ZOLGENSMA
#------------------------------
#------------------------------

results_path_zolg = joinpath(results_path_datainfos, "zolgensma/")
!isdir(results_path_zolg) && mkdir(results_path_zolg)

#------------------------------
# load and preprocess
#------------------------------

# filter for Zolgensma only patients
therapies = sort(unique(baseline_total_df[:,:drug_sma]))
zolg_therapies = ["Monotherapie Zolgensma", therapies[findall(x -> startswith(x, "zolg"), lowercase.(therapies))]...]
zolg_baseline_df = filter(x -> x.drug_sma ∈ zolg_therapies, baseline_total_df)
zolg_ids = unique(zolg_baseline_df[:,:patient_id])
zolg_timedepend_df = filter(x -> x.patient_id ∈ zolg_ids, timedepend_total_df)

# get data for specific tests
test1="hfmse"
test2="rulm"
mixeddata_zolg = get_SMArtCARE_data_one_test(zolg_timedepend_df, zolg_baseline_df, other_vars, baseline_total_vars; test1=test1, test2=test2, remove_lessthan1=true); # 76

# get patient ids 
final_ids_zolg = Int.(mixeddata_zolg.ids)

# extended preprocessing to get timedepend_select_df with correct ids 
testname1, testname2 = get_test_name(test1), get_test_name(test2)
notest1inds = findall(x->ismissing(x),zolg_timedepend_df[:,testname1])
notest2inds = findall(x->ismissing(x),zolg_timedepend_df[:,testname2])
zolg_timedepend_select_df=zolg_timedepend_df[Not(intersect(notest1inds, notest2inds)),:]
test1_vars = get_test_variables(test1, names(zolg_timedepend_select_df))
test2_vars = get_test_variables(test2, names(zolg_timedepend_select_df))
select_vars = vcat(other_vars, test1_vars, test2_vars)
zolg_timedepend_select_df=select(zolg_timedepend_select_df,select_vars)
# filter to ids after preprocessing
zolg_timedepend_select_df = filter(x -> x.patient_id ∈ final_ids_zolg, zolg_timedepend_select_df) 

zolg_baseline_df = filter(x -> x.patient_id ∈ final_ids_zolg, zolg_baseline_df)
age_at_first_treatment_zolg = zolg_baseline_df[:,:age_at_first_treatment]
presymptomatic_zolg = sum(zolg_baseline_df[:,:onset_presymptomatic])/nrow(zolg_baseline_df)

#------------------------------
# basic statistics: create and save patients summary 
#------------------------------

summary_df_zolg = create_patients_and_visits_summary(mixeddata_zolg)
CSV.write(joinpath(results_path_zolg, "summary_patients_and_visits.csv"), summary_df_zolg)

#------------------------------
# visualize typical patterns
#------------------------------

plot_df_zolg = zolg_timedepend_select_df[:,[:patient_id, :test_hfmse, :test_rulm, :months_since_1st_test]]
seed = 65
viz_test = plot_visit_patterns(plot_df_zolg, final_ids_zolg; seed=seed)
savefig(viz_test, joinpath(results_path_zolg, "visits_per_patient_sample_$(seed).pdf"))

#------------------------------
# compare age distribution for Zolgensma and Nusinersen
#------------------------------

age_at_first_treatment_df = DataFrame(
    Statistic = [],
    Nusinersen = [],
    Zolgensma = []
)

push!(age_at_first_treatment_df, ["Mean", mean(age_at_first_treatment_nusi), mean(age_at_first_treatment_zolg)])
push!(age_at_first_treatment_df, ["Standard deviation", std(age_at_first_treatment_nusi), std(age_at_first_treatment_zolg)])
push!(age_at_first_treatment_df, ["1st quartile", quantile(age_at_first_treatment_nusi, 0.25), quantile(age_at_first_treatment_zolg, 0.25)])
push!(age_at_first_treatment_df, ["Median", median(age_at_first_treatment_nusi), median(age_at_first_treatment_zolg)])
push!(age_at_first_treatment_df, ["3rd quartile", quantile(age_at_first_treatment_nusi, 0.75), quantile(age_at_first_treatment_zolg, 0.75)])

CSV.write(joinpath(results_path_datainfos, "age_at_first_treatment.csv"), age_at_first_treatment_df)

#------------------------------
#------------------------------
# RISDIPLAM
#------------------------------
#------------------------------

results_path_risdi = joinpath(results_path_datainfos, "risdiplam/")
!isdir(results_path_risdi) && mkdir(results_path_risdi)

#------------------------------
# load and preprocess
#------------------------------

# filter for Risdiplam patients 
#risdi_baseline_df = filter(x -> x.drug_sma ∈ ["Monotherapie Risdiplam", "Risdi_Zolg_switcher", "Risdi_Nusi_switcher"], baseline_total_df)
risdi_baseline_df = filter(x -> x.drug_sma ∈ ["Monotherapie Risdiplam"], baseline_total_df)
risdi_ids = unique(risdi_baseline_df[:,:patient_id])
risdi_timedepend_df = filter(x -> x.patient_id ∈ risdi_ids, timedepend_total_df)

# get data for specific tests
test1="hfmse"
test2="rulm"
mixeddata_risdi = get_SMArtCARE_data_one_test(risdi_timedepend_df, risdi_baseline_df, other_vars, baseline_total_vars; test1=test1, test2=test2, remove_lessthan1=true); # 154

# get patient ids 
final_ids_risdi = Int.(mixeddata_risdi.ids)

# extended preprocessing to get timedepend_select_df with correct ids 
testname1, testname2 = get_test_name(test1), get_test_name(test2)
notest1inds = findall(x->ismissing(x),risdi_timedepend_df[:,testname1])
notest2inds = findall(x->ismissing(x),risdi_timedepend_df[:,testname2])
risdi_timedepend_select_df=risdi_timedepend_df[Not(intersect(notest1inds, notest2inds)),:]
test1_vars = get_test_variables(test1, names(risdi_timedepend_select_df))
test2_vars = get_test_variables(test2, names(risdi_timedepend_select_df))
select_vars = vcat(other_vars, test1_vars, test2_vars)
risdi_timedepend_select_df=select(risdi_timedepend_select_df,select_vars)
# filter to ids after preprocessing
risdi_timedepend_select_df = filter(x -> x.patient_id ∈ final_ids_risdi, risdi_timedepend_select_df) 

#------------------------------
# basic statistics: create and save patients summary 
#------------------------------

summary_df_risdi = create_patients_and_visits_summary(mixeddata_risdi)
CSV.write(joinpath(results_path_risdi, "summary_patients_and_visits.csv"), summary_df_risdi)

#------------------------------
# visualize typical patterns
#------------------------------

plot_df_risdi = risdi_timedepend_select_df[:,[:patient_id, :test_hfmse, :test_rulm, :months_since_1st_test]]
seed = 65
viz_test = plot_visit_patterns(plot_df_risdi, final_ids_risdi; seed=seed)
savefig(viz_test, joinpath(results_path_risdi, "visits_per_patient_sample_$(seed).pdf"))