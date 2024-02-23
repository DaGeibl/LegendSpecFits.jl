
"""
    f_optimize_ctc(fct, e, qdrift, bin_width)

Calculate the ratio of the FWHM and the peak height of the peak around `peak` in
`e` with a cut window of `window`. The drift time dependence is given by
`e_ctc` = `e` + `fct` * `qdrift`.

# Returns 
    * `fwhm / p_height`: FWHM of the peak divided by peak height
"""
function f_optimize_ctc(fct, e, qdrift, bin_width)
    # calculate drift time corrected energy
    e_ctc = e .+ fct .* qdrift
    # fit peak
    h = fit(Histogram, e_ctc, minimum(e_ctc):bin_width:maximum(e_ctc))
    ps = estimate_single_peak_stats(h)
    result_peak, report_peak = fit_single_peak_th228(h, ps; uncertainty=false)
    # get fwhm and peak height
    fwhm = mvalue(result_peak.fwhm)
    p_height = maximum(report_peak.f_fit.(mvalue(result_peak.μ-0.2*result_peak.σ):0.01:mvalue(result_peak.μ+0.2*result_peak.σ)))
    # use ratio of fwhm and peak height as optimization functional
    return log(fwhm/p_height)
end

"""
    ctc_energy(e::Array{T}, qdrift::Array{T}, peak::T, window::T) where T<:Real

Correct for the drift time dependence of the energy by minimizing the ratio of
the FWHM and the peak height of the peak around `peak` in `e` with a cut window
of `window`. The drift time dependence is given by `qdrift`.

# Returns 
    * `peak`: peak position
    * `window`: window size
    * `fct`: correction factor
    * `bin_width`: optimal bin width
"""
function ctc_energy(e::Array{<:Unitful.Energy{<:Real}}, qdrift::Array{<:Real}, peak::Unitful.Energy{<:Real}, window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}})
    # create cut window around peak
    cut = peak - first(window) .< e .< peak + last(window)
    e_cut, qdrift_cut = e[cut], qdrift[cut]
    e_unit = u"keV"
    # calculate optimal bin width
    bin_width_window = 5.0u"keV"
    bin_width        = get_friedman_diaconis_bin_width(e[peak - bin_width_window .< e .< peak + bin_width_window])
    bin_width_qdrift = get_friedman_diaconis_bin_width(qdrift[peak - bin_width_window .< e .< peak + bin_width_window])

    # get FWHM before correction
    # fit peak
    h_before = fit(Histogram, ustrip.(e_unit, e_cut), ustrip(e_unit, minimum(e_cut)):ustrip(e_unit, bin_width):ustrip(e_unit, maximum(e_cut)))
    ps_before = estimate_single_peak_stats(h_before)
    result_before, report_before = fit_single_peak_th228(h_before, ps_before; uncertainty=true)

    # create function to minimize
    f_minimize = let f_optimize=f_optimize_ctc, e=ustrip.(e_unit, e_cut), qdrift=qdrift_cut, bin_width=ustrip(e_unit, bin_width)
        fct -> f_optimize(fct, e, qdrift, bin_width)
    end

    # minimize function
    fct_range, fct_start = [0.0, 1e-3]*e_unit, [1e-7]*e_unit
    opt_r = optimize(f_minimize, ustrip.(e_unit, fct_range)..., ustrip.(e_unit, fct_start), Fminbox(GradientDescent()), Optim.Options(iterations=1000, show_trace=false, time_limit=600))
    # get optimal correction factor
    fct = Optim.minimizer(opt_r)[1]*e_unit

    # calculate drift time corrected energy
    e_ctc = e_cut .+ fct .* qdrift_cut
    # get FWHM after correction
    # fit peak
    h_after = fit(Histogram, ustrip.(e_unit, e_ctc), ustrip(e_unit, minimum(e_ctc)):ustrip(e_unit, bin_width):ustrip(e_unit, maximum(e_ctc)))
    ps_after = estimate_single_peak_stats(h_after)
    result_after, report_after = fit_single_peak_th228(h_after, ps_after; uncertainty=true)
    result = (
        peak = peak,
        window = window,
        fct = fct,
        bin_width = bin_width,
        bin_width_qdrift = bin_width_qdrift,
        fwhm_before = result_before.fwhm*e_unit,
        fwhm_after = result_after.fwhm*e_unit,
    )
    report = (
        peak = result.peak,
        window = result.window,
        fct = result.fct,
        bin_width = result.bin_width,
        bin_width_qdrift = result.bin_width_qdrift,
        e_peak = e_cut,
        e_ctc = e_ctc, 
        qdrift_peak = qdrift_cut,
        h_before = h_before,
        h_after = h_after,
        fwhm_before = result_before.fwhm,
        fwhm_after = result_after.fwhm,
        report_before = report_before,
        report_after = report_after
    )
    return result, report
end
export ctc_energy