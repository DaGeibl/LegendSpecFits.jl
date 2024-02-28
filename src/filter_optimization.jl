"""
    fit_enc_sigmas(enc_grid::Matrix{T}, enc_grid_rt::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}, min_enc::T, max_enc::T, nbins::Int64, rel_cut_fit::T) where T<:Real

Fit the ENC values in `enc_grid` for each RT in `enc_grid_rt` with a Gaussian and return the optimal RT and the corresponding ENC value.

# Arguments
- `enc_grid`: 2D array of ENC values for each RT in `enc_grid_rt`
- `enc_grid_rt`: 1D array of RT values for which the ENC values in `enc_grid` are calculated
- `min_enc`: minimum ENC value to consider for the fit
- `max_enc`: maximum ENC value to consider for the fit
- `nbins`: number of bins to use for the histogram of ENC values
- `rel_cut_fit`: relative cut value to use for the fit

# Returns
- `rt`: optimal RT value
- `min_enc`: corresponding ENC value
"""
function fit_enc_sigmas(enc_grid::Matrix{T}, enc_grid_rt::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}, min_enc::T, max_enc::T, nbins::Int64, rel_cut_fit::T) where T<:Real
    @assert size(enc_grid, 1) == length(enc_grid_rt) "enc_grid and enc_grid_rt must have the same number of columns"
    
    # create empty array for results
    rts = Quantity{<:Real}[]
    enc = Measurement[]
    
    for (r, rt) in enumerate(enc_grid_rt)
        # get enc for this rt
        enc_rt = flatview(enc_grid)[r, :]
        # sanity check
        if all(enc_rt .== 0.0)
            continue
        end
        # get cut value
        cuts = cut_single_peak(enc_rt, min_enc, max_enc,; n_bins=nbins, relative_cut=rel_cut_fit)

        # fit gaussian
        result, _ = fit_single_trunc_gauss(enc_rt, cuts)

        # get sigma
        push!(enc, result.σ)
        push!(rts, rt)
    end

    # get minimal enc and rt
    min_enc     = minimum(enc)
    rt_min_enc  = rts[findmin(enc)[2]]
    
    # generate result and report
    result = (
        rt = measurement(rt_min_enc, step(enc_grid_rt)),
        min_enc = min_enc
    )
    report = (
        rt = result.rt,
        min_enc = result.min_enc,
        enc_grid_rt = rts,
        enc = enc
    )
    return result, report

end
export fit_enc_sigmas

"""
fit_fwhm_ft_fep(e_grid::Matrix, e_grid_ft::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, nbins::Int64, rel_cut_fit::T; default_ft::Quantity{T}=3.0u"µs") where {T <:Real}

Fit the FWHM values in `e_grid` for each FT in `e_grid_ft` with a Gamma Peakshape and return the optimal FT and the corresponding FWHM value. The cut values cut for each flat-top time a window for better histogramming.

# Arguments
- `e_grid`: 2D array of energy values for each FT in `e_grid_ft`
- `e_grid_ft`: 1D array of FT values for which the FWHM values in `e_grid` are calculated
- `rt`: RT value for which the FWHM values in `e_grid` are calculated
- `min_e`: minimum energy value to consider for the fit
- `max_e`: maximum energy value to consider for the fit
- `nbins`: number of bins to use for the histogram of energy values
- `rel_cut_fit`: relative cut value to use for the fit

# Returns
- `ft`: optimal FT value
- `min_fwhm`: corresponding FWHM value
"""
function fit_fwhm_ft_fep(e_grid::Matrix, e_grid_ft::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, nbins::Int64, rel_cut_fit::T; default_ft::Quantity{T}=3.0u"µs") where {T <:Real}
    @assert size(e_grid, 1) == length(e_grid_ft) "e_grid and e_grid_rt must have the same number of columns"
    
    # create empty array for results
    fts  = Quantity{<:Real}[]
    fwhm = Measurement[]
    
    for (r, ft) in enumerate(e_grid_ft)
        # if ft > rt filter doesn't make sense, continue
        if ft > rt
            @debug "FT $ft bigger than RT $rt, skipping"
            continue
        end
        # get e values for this ft
        e_ft = Array{Float64}(flatview(e_grid)[r, :])
        e_ft = e_ft[isfinite.(e_ft)]

        # sanity check
        if count(min_e .< e_ft .< max_e) < 100
            @debug "Not enough data points for FT $ft, skipping"
            continue
        end
        # cut around peak to increase performance
		fit_cut = cut_single_peak(e_ft, min_e, max_e,; n_bins=nbins, relative_cut=rel_cut_fit)
		e_ft = e_ft[fit_cut.max - 300 .< e_ft .< fit_cut.max + 300]

        # create histogram from it
        bin_width = 2 * (quantile(e_ft, 0.75) - quantile(e_ft, 0.25)) / ∛(length(e_ft))
        h = fit(Histogram, e_ft, minimum(e_ft):bin_width:maximum(e_ft))

        # create peakstats
        ps = estimate_single_peak_stats_th228(h)
        # check if ps guess is valid
        if any(tuple_to_array(ps) .<= 0)
            @debug "Invalid guess for peakstats, skipping"
            continue
        end
        # fit peak 
        result, _ = fit_single_peak_th228(h, ps,; uncertainty=false)
        # get fwhm
        push!(fwhm, result.fwhm)
        push!(fts, ft)
    end

    # get minimal fwhm and rt
    if isempty(fwhm)
        @warn "No valid FWHM found, setting to NaN"
        min_fwhm = NaN * u"keV"
        @warn "No valid FT found, setting to default"
        ft_min_fwhm = default_ft
    else
        # calibration constant from last fit to get rough calibration for better plotting
        c = 2614.5u"keV" ./ result.μ
        fwhm = fwhm .* c 
        # get minimal fwhm and ft
        min_fwhm    = minimum(fwhm)
        ft_min_fwhm = e_grid_ft[findmin(fwhm)[2]]
    end
    # generate result and report
    result = (
        ft = measurement(ft_min_fwhm, step(e_grid_ft)),
        min_fwhm = min_fwhm
    )
    report = (
        ft = result.ft, 
        min_fwhm = result.min_fwhm,
        e_grid_ft = fts,
        fwhm = fwhm,
    )
    return result, report

end
export fit_fwhm_ft_fep

"""
    fit_sg_wl(dep_sep_data, a_grid_wl_sg, optimization_config)

Fit the SG window length for the SEP data and return the optimal window length and the corresponding survival fraction.

# Arguments
- `dep_sep_data`: NamedTuple with the DEP and SEP data
- `a_grid_wl_sg`: range of window lengths to sweep through
- `optimization_config`: configuration dictionary

# Returns
- `result`: optimal window length and corresponding survival fraction
- `report`: report with all window lengths and survival fractions
"""
function fit_sg_wl(dep_sep_data::NamedTuple{(:dep, :sep)}, a_grid_wl_sg::StepRangeLen, optimization_config::PropDict)
    # unpack config
    dep, dep_window = optimization_config.dep, Float64.(optimization_config.dep_window)
    sep, sep_window = optimization_config.sep, Float64.(optimization_config.sep_window)

    # unpack data
    e_dep, e_sep = dep_sep_data.dep.e, dep_sep_data.sep.e
    aoe_dep, aoe_sep = dep_sep_data.dep.aoe, dep_sep_data.sep.aoe

    # prepare peakhist
    result_dep, _ = prepare_dep_peakhist(e_dep, dep; n_bins_cut=optimization_config.nbins_dep_cut, relative_cut=optimization_config.dep_rel_cut, uncertainty=false)
    
    yield()
    
    # get calib constant from fit on DEP peak
    e_dep_calib = e_dep .* mvalue(result_dep.m_calib)
    e_sep_calib = e_sep .* mvalue(result_dep.m_calib)

    # create empty arrays for sf and sf_err
    sep_sfs     = Quantity{Measurement}[]
    wls        = Quantity{<:Real}[]

    # for each window lenght, calculate the survival fraction in the SEP
    for (i_aoe, wl) in enumerate(a_grid_wl_sg)

        aoe_dep_i = aoe_dep[i_aoe, :][isfinite.(aoe_dep[i_aoe, :])] ./ mvalue(result_dep.m_calib)
        e_dep_i   = e_dep_calib[isfinite.(aoe_dep[i_aoe, :])]

        # prepare AoE
        max_aoe_dep_i = quantile(aoe_dep_i, optimization_config.max_aoe_quantile) + optimization_config.max_aoe_offset
        min_aoe_dep_i = quantile(aoe_dep_i, optimization_config.min_aoe_quantile) + optimization_config.min_aoe_offset

        try
            psd_cut = get_psd_cut(aoe_dep_i, e_dep_i; window=dep_window, cut_search_interval=(min_aoe_dep_i, max_aoe_dep_i), uncertainty=false)

            aoe_sep_i = aoe_sep[i_aoe, :][isfinite.(aoe_sep[i_aoe, :])] ./ result_dep.m_calib
            e_sep_i   = e_sep_calib[isfinite.(aoe_sep[i_aoe, :])]

            result_sep, _ = get_peak_surrival_fraction(aoe_sep_i, e_sep_i, sep, sep_window, psd_cut.cut; uncertainty=false, low_e_tail=false)

            push!(sep_sfs, result_sep.sf)
            push!(wls, wl)
        catch e
            @warn "Couldn't process window length $wl"
        end
        yield()
    end
    # get minimal surrival fraction and window length
    sep_sfs_cut = 1.0u"percent" .< sep_sfs .< 100u"percent"
    if isempty(sep_sfs[sep_sfs_cut])
        @warn "No valid SEP SF found, setting to NaN"
        min_sf = measurement(NaN, NaN)*u"percent"
        @warn "No valid window length found, setting to default"
        wl_sg_min_sf = last(a_grid_wl_sg[a_grid_wl_sg .< 110u"ns"])
    else
        min_sf       = minimum(sep_sfs[sep_sfs_cut])
        wl_sg_min_sf = wls[sep_sfs_cut][findmin(sep_sfs[sep_sfs_cut])[2]]
    end

    # generate result and report
    result = (
        wl = measurement(wl_sg_min_sf, step(a_grid_wl_sg)),
        sf = min_sf,
        n_dep = length(e_dep),
        n_sep = length(e_sep)
    )
    report = (
        wl = result.wl,
        min_sf = result.sf,
        a_grid_wl_sg = wls,
        sfs = sep_sfs
    )
    return result, report
end
export fit_sg_wl