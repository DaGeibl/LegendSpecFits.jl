"""
    ctc_lq(lq_all::Array{T}, ecal_all::Array{T}, qdrift_e_all::Array{T}, compton_bands::Array{T}, peak::T, window::T) where T<:Real

Correct for the drift time dependence of the LQ parameter

# Returns (but may change):
    * `peak`: peak position
    * `window`: window size
    * `fct`: correction factor
    * `σ_before`: σ before correction
    * `σ_after`: σ after correction
    * `func`: function to correct lq
    * `func_generic`: generic function to correct lq
"""


function ctc_lq(lq::Vector{<:Real}, e::Vector{<:Unitful.RealOrRealQuantity}, qdrift::Vector{<:Real}, dep_µ::Unitful.AbstractQuantity, dep_σ::Unitful.AbstractQuantity;
    hist_start::Real = -0.5, hist_end::Real = 2.5, bin_width::Real = 0.01, relative_cut::Float64 = 0.4, 
    ctc_dep_edgesigma::Float64=3.0, lq_expression::Union{Symbol, String}="lq / e", qdrift_expression::Union{Symbol, String} = "qdrift / e", pol_order::Int=1)

    # calculate DEP edges
    dep_left = dep_µ - ctc_dep_edgesigma * dep_σ
    dep_right = dep_µ + ctc_dep_edgesigma * dep_σ


    # cut lq values from dep
    cut = dep_left .< e .< dep_right .&& isfinite.(lq) .&& isfinite.(qdrift)
    @debug "Cut window: $(dep_left) < e < $(dep_right)"
    lq_cut, qdrift_cut = lq[cut], qdrift[cut]

    h_before = fit(Histogram, lq_cut, hist_start:bin_width:hist_end)

    # get σ before correction
    # fit peak
    cut_peak = cut_single_peak(lq_cut, -10, 20; n_bins=-1, relative_cut)
    println("cut_peak: ", cut_peak)
    result_before, report_before = fit_single_trunc_gauss(lq_cut, cut_peak; uncertainty=false)
    @debug "Found Best σ before correction: $(round(result_before.σ, digits=2))"

    # create optimization function
    function f_optimize_ctc(fct, lq_cut, qdrift_cut)
        # calculate drift time corrected lq
        lq_ctc =  lq_cut .+ PolCalFunc(0.0, fct...).(qdrift_cut)
        # fit peak
        cuts_lq = cut_single_peak(lq_ctc, -10, 20; n_bins=-1, relative_cut)
        result_after, _ = fit_single_trunc_gauss(lq_ctc, cuts_lq; uncertainty=false)
        return mvalue(result_after.σ)
    end

    # create function to minimize
    f_minimize = let f_optimize=f_optimize_ctc, lq_cut=lq_cut, qdrift_cut=qdrift_cut
        fct -> f_optimize(fct, lq_cut, qdrift_cut)
    end

    qdrift_median = median(qdrift_cut)
    # lower bound
    #fct_lb = [- 0.1^i for i in 1:pol_order]
    fct_lb = [-(1 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Lower bound: $fct_lb"
    # upper bound
    #fct_ub = [0.1^i for i in 1:pol_order]
    fct_ub = [(0.1 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Upper bound: $fct_ub"
    # start value
    #fct_start = [0.0 for i in 1:pol_order]
    fct_start = [-(0.001 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Start value: $fct_start"

    opt_bounds = (lower = fct_lb, upper = fct_ub, start = fct_start, qdrift_median = qdrift_median)

    println("fct_lb: ", fct_lb)
    println("fct_ub: ", fct_ub)
    println("fct_start: ", fct_start)

    # optimization
    optf = OptimizationFunction((u, p) -> f_minimize(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, fct_start, (), lb=fct_lb, ub=fct_ub)
    res = solve(optpro, NLopt.LN_BOBYQA(), maxiters = 3000, maxtime=optim_time_limit)
    converged = (res.retcode == ReturnCode.Success)

    # get optimal correction factor
    fct = res.u
    @debug "Found Best FCT: $(fct .* 1e6)E-6"

    if !converged @warn "CTC did not converge" end

    # calculate drift time corrected lq
    lq_ctc_corrected = lq_cut .+ PolCalFunc(0.0, fct...).(qdrift_cut)
    
    # normalize once again to μ = 0 and σ = 1
    h_after = fit(Histogram, lq_ctc_corrected, hist_start:bin_width:hist_end)
    
    _cuts_lq = cut_single_peak(lq_ctc_corrected, hist_start, eltype(hist_start)(10); n_bins=-1, relative_cut)
    result_after, report_after = fit_single_trunc_gauss(lq_ctc_corrected, _cuts_lq, uncertainty=true)
    μ_norm = mvalue(result_after.μ)
    σ_norm = mvalue(result_after.σ)
    lq_ctc_normalized = (lq_ctc_corrected .- μ_norm) ./ σ_norm

    # get cal PropertyFunctions
    lq_ctc_func = "( ( $(lq_expression) ) + " * join(["$(fct[i]) * ( $(qdrift_expression) )^$(i)" for i in eachindex(fct)], " + ") * " - $(μ_norm) ) / $(σ_norm) "

    # create final histograms after normalization
    cuts_lq = cut_single_peak(lq_ctc_normalized, hist_start, eltype(hist_start)(10); n_bins=-1, relative_cut)
    result_after_norm, report_after_norm = fit_single_trunc_gauss(lq_ctc_normalized, cuts_lq, uncertainty=true)

    h_after_norm = fit(Histogram, lq_ctc_normalized, -5:bin_width:10)

    result = (
        dep_left = dep_left,
        dep_right = dep_right,
        opt_bounds = opt_bounds,
        func = lq_ctc_func,
        fct = fct,
        σ_start = f_minimize(0.0),
        σ_optimal = f_minimize(fct),
        σ_before = result_before.σ,
        σ_after = result_after.σ,
        σ_after_norm = result_after_norm.σ,
        before = result_before,
        after = result_after,
        after_norm = result_after_norm,
        converged = converged
    )
    report = (
        dep_left = dep_left,
        dep_right = dep_right,
        opt_bounds = opt_bounds,
        fct = result.fct,
        bin_width = bin_width,
        lq_peak = lq_cut,
        lq_ctc_corrected = lq_ctc_corrected, 
        lq_ctc_normalized = lq_ctc_normalized, 
        qdrift_peak = qdrift_cut,
        h_before = h_before,
        h_after = h_after,
        h_after_norm = h_after_norm,
        σ_before = result.σ_before,
        σ_after = result.σ_after,
        σ_after_norm = result.σ_after_norm,
        report_before = report_before,
        report_after = report_after,
        report_after_norm = report_after_norm
    )
    return result, report
end
export ctc_lq


"""
    lq_ctc_lin_fit(lq::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, dep_µ::Unitful.AbstractQuantity, dep_σ::Unitful.AbstractQuantity; 
    ctc_dep_edgesigma::Float64=3.0, ctc_lq_precut_relative_cut::Float64=0.25, lq_outlier_sigma::Float64 = 2.0, ctc_driftime_cutoff_method::Symbol=:percentile, dt_eff_outlier_sigma::Float64 = 2.0, lq_e_corr_expression::Union{String,Symbol}="lq / e", dt_eff_expression::Union{String,Symbol}="qdrift / e", ctc_dt_eff_low_quantile::Float64=0.15, ctc_dt_eff_high_quantile::Float64=0.95, pol_fit_order::Int=1, uncertainty::Bool=false)

    Perform the drift time correction on the LQ data using the DEP peak. The function cuts outliers in lq and drift time, then performs a polynomial fit on the remaining data. The data is Corrected by subtracting the polynomial fit from the lq data.

# Arguments 
    * `lq`: Energy corrected lq parameter
    * `dt_eff`: Effective drift time
    * `e_cal`: Energy
    * `dep_µ`: Mean of the DEP peak
    * `dep_σ`: Standard deviation of the DEP peak

# Keywords
    * `ctc_dep_edgesigma`: Number of standard deviations used to define the DEP edges
    * `ctc_lq_precut_relative_cut`: Relative cut for cut_single_peak function
    * `ctc_driftime_cutoff_method`: Method used to define the drift time cutoff
    * `lq_outlier_sigma`: Number of standard deviations used to define the lq cutoff
    * `dt_eff_outlier_sigma`: Number of standard deviations used to define the drift time cutoff
    * `lq_e_corr_expression`: Expression for the energy corrected lq classifier 
    * `dt_eff_expression`: Expression for the effective drift time 
    * `ctc_dt_eff_low_quantile`: Lower quantile used to define the drift time cutoff
    * `ctc_dt_eff_high_quantile`: Higher quantile used to define the drift time cutoff
    * `pol_fit_order`: Order of the polynomial fit used for the drift time correction

# Returns
    * `result`: NamedTuple of the function used for the drift time correction, the polynomial fit result and the box constraints
    * `report`: NamedTuple of the histograms used for the fit, the cutoff values and the DEP edges

"""
function lq_ctc_lin_fit(
    lq::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, dep_µ::Unitful.AbstractQuantity, dep_σ::Unitful.AbstractQuantity; 
    ctc_dep_edgesigma::Float64=3.0, ctc_lq_precut_relative_cut::Float64=0.25, lq_outlier_sigma::Float64 = 2.0, ctc_driftime_cutoff_method::Symbol=:percentile, dt_eff_outlier_sigma::Float64 = 2.0, lq_e_corr_expression::Union{String,Symbol}="lq / e", dt_eff_expression::Union{String,Symbol}="qdrift / e", ctc_dt_eff_low_quantile::Float64=0.15, ctc_dt_eff_high_quantile::Float64=0.95, pol_fit_order::Int=1, uncertainty::Bool=false) 

    # calculate DEP edges
    dep_left = dep_µ - ctc_dep_edgesigma * dep_σ
    dep_right = dep_µ + ctc_dep_edgesigma * dep_σ

    # cut data to DEP peak
    dep_finite = (dep_left .< e_cal .< dep_right .&& isfinite.(lq) .&& isfinite.(dt_eff))
    lq_dep = lq[dep_finite]
    dt_eff_dep = ustrip.(dt_eff[dep_finite])
   
    # precut lq data for fit
    lq_precut = cut_single_peak(lq_dep, minimum(lq_dep), quantile(lq_dep, 0.99); relative_cut=ctc_lq_precut_relative_cut)

    # truncated gaussian fit
    lq_result, lq_report = fit_single_trunc_gauss(lq_dep, lq_precut; uncertainty)
    µ_lq = mvalue(lq_result.μ)
    σ_lq = mvalue(lq_result.σ)

    # set cutoff in lq dimension for later fit
    lq_lower = µ_lq - lq_outlier_sigma * σ_lq 
    lq_upper = µ_lq + lq_outlier_sigma * σ_lq 


    # dt_eff_dep cutoff; method dependant on detector type
    
    if ctc_driftime_cutoff_method == :percentile # standard method; can be used for all detectors
        #set cutoff; default at the 15% and 95% percentile
        t_lower = quantile(dt_eff_dep, ctc_dt_eff_low_quantile)
        t_upper = quantile(dt_eff_dep, ctc_dt_eff_high_quantile)
        drift_report = nothing

    elseif ctc_driftime_cutoff_method == :gaussian # can't be used for detectors with double peaks
        
        dt_eff_precut = cut_single_peak(dt_eff_dep, minimum(lq_dep), maximum(lq_dep))
        drift_result, drift_report = fit_single_trunc_gauss(dt_eff_dep, dt_eff_precut; uncertainty)
        µ_t = mvalue(drift_result.μ)
        σ_t = mvalue(drift_result.σ)

        #set cutoff in drift time dimension for later fit
        t_lower = µ_t - dt_eff_outlier_sigma * σ_t
        t_upper = µ_t + dt_eff_outlier_sigma * σ_t

    elseif ctc_driftime_cutoff_method == :double_gaussian # can be used for detectors with double peaks; not optimized yet
        #create histogram for drift time
        ideal_length = get_number_of_bins(dt_eff_dep)
        drift_prehist = fit(Histogram, dt_eff_dep, range(minimum(dt_eff_dep), stop=maximum(dt_eff_dep), length=ideal_length))
        drift_prestats = estimate_single_peak_stats(drift_prehist)

        #fit histogram with double gaussian
        drift_result, drift_report = fit_binned_double_gauss(drift_prehist, drift_prestats; uncertainty)
        
        #set cutoff at the x-value where the fit function is 10% of its maximum value
        x_values = -1000:0.5:5000  
        max_value = maximum(drift_report.f_fit.(x_values))
        threshold = 0.1 * max_value

        g(x) = drift_report.f_fit(x) - threshold
        x_at_threshold = find_zeros(g, minimum(x_values), maximum(x_values))

        t_lower = minimum(x_at_threshold)
        t_upper = maximum(x_at_threshold)
    else
        throw(ArgumentError("Drift time cutoff method $ctc_driftime_cutoff_method not supported"))
    end

    # store cutoff values in box to return later    
    box = (;lq_lower, lq_upper, t_lower, t_upper)

    # cut data according to cutoff values
    lq_cut = lq_dep[lq_lower .< lq_dep .< lq_upper .&& t_lower .< dt_eff_dep .< t_upper]
    t_cut = dt_eff_dep[lq_lower .< lq_dep .< lq_upper .&& t_lower .< dt_eff_dep .< t_upper]

    # polynomial fit
    result_µ, report_µ = chi2fit(pol_fit_order, t_cut, lq_cut; uncertainty)
    par = mvalue(result_µ.par)
    pol_fit_func = report_µ.f_fit

    # property function for drift time correction
    lq_class_func = "( $lq_e_corr_expression ) - " * join(["$(par[i]) * ($dt_eff_expression)^$(i-1)" for i in eachindex(par)], " - ")
    lq_class_func_generic = "( lq / e ) - (slope * qdrift / e + y_inter)"

    # create result and report
    result = (
    func = lq_class_func,
    func_generic = lq_class_func_generic,
    fit_result = result_µ,
    box_constraints = box,
    )

    report = (
    lq_report = lq_report, 
    drift_report = drift_report,
    lq_box = box,
    drift_time_func = pol_fit_func,
    dep_left = dep_left,
    dep_right = dep_right,
    )
    

    return result, report
end
export lq_ctc_lin_fit

