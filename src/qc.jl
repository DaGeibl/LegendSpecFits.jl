

"""
    qc_sg_optimization(dsp_dep, dsp_sep, optimization_config)

Perform simple QC cuts on the DEP and SEP data and return the data for the optimization of the SG window length.
"""
function qc_sg_optimization(dsp_dep::NamedTuple{(:aoe, :e, :blmean, :blslope, :t50)}, dsp_sep::NamedTuple{(:aoe, :e, :blmean, :blslope, :t50)}, optimization_config::PropDict)
    ### DEP
    # Load DEP data and prepare Pile-up cut
    blslope_dep, t50_dep = dsp_dep.blslope[isfinite.(dsp_dep.e)], dsp_dep.t50[isfinite.(dsp_dep.e)]
    aoe_dep, e_dep = dsp_dep.aoe[:, isfinite.(dsp_dep.e)], dsp_dep.e[isfinite.(dsp_dep.e)]
    # get half truncated centered cut on blslope for pile-up rejection
    result_dep_slope_cut, report_dep_slope_cut = get_centered_gaussian_window_cut(blslope_dep, -0.1u"ns^-1", 0.1u"ns^-1", optimization_config.cuts.dep.blslope_sigma, ; n_bins_cut=optimization_config.cuts.dep.nbins_blslope_cut, relative_cut=optimization_config.cuts.dep.rel_cut_blslope_cut)
    # Cut on blslope, energy and t0 for simple QC
    qc_cut_dep = blslope_dep .> result_dep_slope_cut.low_cut .&& blslope_dep .< result_dep_slope_cut.high_cut .&& e_dep .> optimization_config.cuts.dep.min_e .&& quantile(e_dep, first(optimization_config.cuts.dep.e_quantile)) .< e_dep .< quantile(e_dep, last(optimization_config.cuts.dep.e_quantile)) .&& first(optimization_config.cuts.dep.t50) .< t50_dep .< last(optimization_config.cuts.dep.t50)
    aoe_dep, e_dep = aoe_dep[:, qc_cut_dep], e_dep[qc_cut_dep]

    ### SEP
    # Load SEP data and prepare Pile-up cut
    blslope_sep, t50_sep = dsp_sep.blslope[isfinite.(dsp_sep.e)], dsp_sep.t50[isfinite.(dsp_sep.e)]
    aoe_sep, e_sep = dsp_sep.aoe[:, isfinite.(dsp_sep.e)], dsp_sep.e[isfinite.(dsp_sep.e)]

    # get half truncated centered cut on blslope for pile-up rejection
    result_sep_slope_cut, report_sep_slope_cut = get_centered_gaussian_window_cut(blslope_sep, -0.1u"ns^-1", 0.1u"ns^-1", optimization_config.cuts.sep.blslope_sigma, ; n_bins_cut=optimization_config.cuts.sep.nbins_blslope_cut, relative_cut=optimization_config.cuts.sep.rel_cut_blslope_cut)

    # Cut on blslope, energy and t0 for simple QC
    qc_cut_sep = blslope_sep .> result_sep_slope_cut.low_cut .&& blslope_sep .< result_sep_slope_cut.high_cut .&& e_sep .> optimization_config.cuts.sep.min_e .&& quantile(e_sep, first(optimization_config.cuts.sep.e_quantile)) .< e_sep .< quantile(e_sep, last(optimization_config.cuts.sep.e_quantile)) .&& first(optimization_config.cuts.sep.t50) .< t50_sep .< last(optimization_config.cuts.sep.t50)
    aoe_sep, e_sep = aoe_sep[:, qc_cut_sep], e_sep[qc_cut_sep]

    return (dep=(aoe=aoe_dep, e=e_dep), sep=(aoe=aoe_sep, e=e_sep))
end
export qc_sg_optimization


"""
    qc_cal_energy(data, qc_config)

Perform simple QC cuts on the data and return the data for energy calibration.
"""
function qc_cal_energy(data::Q, qc_config::PropDict) where Q<:Table
    # get bl mean cut
    result_blmean, _ = get_centered_gaussian_window_cut(data.blmean, qc_config.blmean.min, qc_config.blmean.max, qc_config.blmean.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blmean.n_bins_fraction)), relative_cut=qc_config.blmean.relative_cut, fixed_center=false, left=true)
    blmean_qc = result_blmean.low_cut .< data.blmean .< result_blmean.high_cut
    @debug format("Baseline Mean cut surrival fraction {:.2f}%", count(blmean_qc) / length(data) * 100)
    # get bl slope cut
    result_blslope, _ = get_centered_gaussian_window_cut(data.blslope, qc_config.blslope.min, qc_config.blslope.max, qc_config.blslope.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blslope.n_bins_fraction)), relative_cut=qc_config.blslope.relative_cut, fixed_center=true, left=false, center=zero(data.blslope[1]))
    blslope_qc = result_blslope.low_cut .< data.blslope .< result_blslope.high_cut
    @debug format("Baseline Slope cut surrival fraction {:.2f}%", count(blslope_qc) / length(data) * 100)
    # get blsigma cut
    result_blsigma, _ = get_centered_gaussian_window_cut(data.blsigma, qc_config.blsigma.min, qc_config.blsigma.max, qc_config.blsigma.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blsigma.n_bins_fraction)), relative_cut=qc_config.blsigma.relative_cut, fixed_center=false, left=true)
    blsigma_qc = result_blsigma.low_cut .< data.blsigma .< result_blsigma.high_cut
    @debug format("Baseline Sigma cut surrival fraction {:.2f}%", count(blsigma_qc) / length(data) * 100)
    # get t0 cut
    t0_qc = qc_config.t0.min .< data.t0 .< qc_config.t0.max
    @debug format("t0 cut surrival fraction {:.2f}%", count(t0_qc) / length(data) * 100)
    # get intrace pile-up cut
    inTrace_qc = .!(data.inTrace_intersect .> data.t0 .+ 2 .* data.drift_time .&& data.inTrace_n .> 1)
    @debug format("Intrace pile-up cut surrival fraction {:.2f}%", count(inTrace_qc) / length(data) * 100)
    # get energy cut
    energy_qc = qc_config.e_trap.min .< data.e_trap .&& isfinite.(data.e_trap) .&& isfinite.(data.e_zac) .&& isfinite.(data.e_cusp)
    @debug format("Energy cut surrival fraction {:.2f}%", count(energy_qc) / length(data) * 100)

    # combine all cuts
    qc_tab = TypedTables.Table(blmean = blmean_qc, blslope = blslope_qc, blsigma = blsigma_qc, t0 = t0_qc, inTrace = inTrace_qc, energy = energy_qc, qc = blmean_qc .&& blslope_qc .&& blsigma_qc .&& t0_qc .&& inTrace_qc .&& energy_qc)
    @debug format("Total QC cut surrival fraction {:.2f}%", count(qc) / length(data) * 100)
    return qc_tab, (blmean = result_blmean, blslope = result_blslope, blsigma = result_blsigma)
end
export qc_cal_energy


"""
    pulser_cal_qc(data, pulser_config; n_pulser_identified=100)

Perform simple QC cuts on the data and return the data for energy calibration.
# Returns 
    - pulser_idx: indices of the pulser events
"""
function pulser_cal_qc(data::Q, pulser_config::PropDict; n_pulser_identified::Int=100) where Q<:Table
    # extract config
    f = pulser_config.frequency
    T = upreferred(1/f)
    # get drift time cut
    peakhist, peakpos = RadiationSpectra.peakfinder(fit(Histogram, ustrip.(data.drift_time[pulser_config.drift_time.min .< data.drift_time .< pulser_config.drift_time.max]), ustrip(pulser_config.drift_time.min:pulser_config.drift_time.bin_width:pulser_config.drift_time.max)), σ=ustrip(pulser_config.drift_time.peak_width), backgroundRemove=true, threshold=pulser_config.drift_time.threshold)
    # select peak with highest prominence in background removed histogram
    pulser_drift_time_peak = peakpos[last(findmax([maximum(peakhist.weights[pp-ustrip(pulser_config.drift_time.peak_width) .< first(peakhist.edges)[2:end] .< pp+ustrip(pulser_config.drift_time.peak_width)]) for pp in peakpos]))]*unit(data.drift_time[1])
    # get drift time idx in peak
    drift_time_idx = findall(x -> pulser_drift_time_peak - pulser_config.drift_time.peak_width < x < pulser_drift_time_peak + pulser_config.drift_time.peak_width, data.drift_time)
    ts = data.timestamp[drift_time_idx]
    pulser_identified_idx = findall(x -> x .== T, diff(ts))
    @info "Found pulser peak in drift time distribution at $(pulser_drift_time_peak)"
    if isempty(pulser_identified_idx)
        @warn "No pulser events found in the data, try differen method"
        pulser_identified_idx = findall(x -> T - 10u"ns" < x < T + 10u"ns", diff(ts))
    end
    if isempty(pulser_identified_idx)
        @warn "No pulser events found in the data"
        return Int64[]
    end
    # iterate through different pulser options and return unique idxs
    pulser_idx = Int64[]
    for idx in rand(pulser_identified_idx, n_pulser_identified)
        p_evt = data[drift_time_idx[idx]]
        append!(pulser_idx, findall(pulser_config.pulser_diff.min .< (data.timestamp .- p_evt.timestamp .+ (T/4)) .% (T/2) .- (T/4) .< pulser_config.pulser_diff.max))
    end
    unique!(pulser_idx)
end
export pulser_cal_qc