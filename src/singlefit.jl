"""
    fit_single_trunc_gauss(x::Array, cuts::NamedTuple{(:low, :high, :max), Tuple{Float64, Float64, Float64}})

Fit a single truncated Gaussian to the data `x` between `min_x` and `max_x`.
Returns `report` and `result`` with:
    * `f_fit`: fitted function
    * `μ`: mean of the Gaussian
    * `μ_err`: error of the mean
    * `σ`: standard deviation of the Gaussian
    * `σ_err`: error of the standard deviation
    * `n`: number of counts in the peak
"""
function fit_single_trunc_gauss(x::Vector{<:Unitful.RealOrRealQuantity}, cuts::NamedTuple{(:low, :high, :max), Tuple{<:T, <:T, <:T}}=(low = zero(first(x))*NaN, high = zero(first(x))*NaN, max = zero(first(x))*NaN); uncertainty::Bool=true) where T<:Unitful.RealOrRealQuantity
    @assert unit(cuts.low) == unit(cuts.high) == unit(cuts.max) == unit(x[1]) "Units of min_x, max_x and x must be the same"
    x_unit = unit(x[1])
    x, cut_low, cut_high, cut_max = ustrip.(x), ustrip(cuts.low), ustrip(cuts.high), ustrip(cuts.max)
    cut_low, cut_high = ifelse(isnan(cut_low), minimum(x), cut_low), ifelse(isnan(cut_high), maximum(x), cut_high)

    bin_width = get_friedman_diaconis_bin_width(x[(x .> cut_low) .&& (x .< cut_high)])
    x_min, x_max = minimum(x), maximum(x)
    h_nocut = fit(Histogram, x, x_min:bin_width:x_max)
    ps = estimate_single_peak_stats_th228(h_nocut)
    @debug "Peak stats: $ps"

    # cut peak out of data
    x = x[(x .> cut_low) .&& (x .< cut_high)]
    h = fit(Histogram, x, cut_low:bin_width:cut_high)
    n = length(x)

    # create fit functions
    f_fit(x, v) = pdf(truncated(Normal(v.μ, v.σ), cut_low, cut_high), x)
    f_fit_n(x, v) = n * f_fit(x, v)
    
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μ = Normal(ps.peak_pos, ps.peak_sigma/4),
        σ = weibull_from_mx(ps.peak_sigma, 1.5*ps.peak_sigma),
    )

    # create fit model
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    
    v_init  = mean(pseudo_prior)

    f_loglike = let cut_low = cut_low, cut_high = cut_high, x = x
        v -> (-1) * loglikelihood(truncated(Normal(v[1], v[2]), cut_low, cut_high), x)
    end

    # MLE
    opt_r = optimize(f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(opt_r.minimizer)

    if uncertainty
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance = inv(H)

        param_covariance = nothing
        if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)
        
        # TODO: p-values etc for unbinned fits
        # get p-value
        pval, chi2, dof = p_value(f_fit_n, h, v_ml)
        
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_fit_n, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) * x_unit for k in keys(v_ml)]...),
                  (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                  residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"

        result = merge(v_ml, )
    end

    # normalize nocut histogram to PDF of cut histogram
    h_pdf = Histogram(h_nocut.edges[1], h_nocut.weights ./ sum(h.weights) ./ step(h.edges[1]))

    report = (
        f_fit = t -> Base.Fix2(f_fit, v_ml)(t),
        h = h_pdf,
        μ = result.μ,
        σ = result.σ,
        gof = get(result, :gof, NamedTuple())
    )
    return (result = result, report = report)
end
export fit_single_trunc_gauss

"""
    fit_half_centered_trunc_gauss(x::Array, cuts::NamedTuple{(:low, :high, :max), Tuple{Float64, Float64, Float64}})
Fit a single truncated Gaussian to the data `x` between `cut.low` and `cut.high`. The peak center is fixed at `μ` and the peak is cut in half either in the left or right half.
# Returns `report` and `result`` with:
    * `f_fit`: fitted function
    * `μ`: mean of the Gaussian
    * `σ`: standard deviation of the Gaussian
"""
function fit_half_centered_trunc_gauss(x::Vector{<:Unitful.RealOrRealQuantity}, μ::T, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}; left::Bool=false, uncertainty::Bool=true) where T<:Unitful.RealOrRealQuantity
    @assert unit(cuts.low) == unit(cuts.high) == unit(cuts.max) == unit(x[1]) == unit(μ) "Units of min_x, max_x and x must be the same"
    x_unit = unit(x[1])
    x, cut_low, cut_high, cut_max, μ = ustrip.(x), ustrip(cuts.low), ustrip(cuts.high), ustrip(cuts.max), ustrip(μ)

    # get peak stats
    bin_width = get_friedman_diaconis_bin_width(x[(x .> cut_low) .&& (x .< cut_high)])
    x_min, x_max = minimum(x), maximum(x)
    h_nocut = fit(Histogram, x, x_min:bin_width:x_max)
    ps = estimate_single_peak_stats_th228(h_nocut)
    @debug "Peak stats: $ps"

    # cut peak out of data
    x = ifelse(left, x[(x .> cut_low) .&& (x .< cut_high) .&& x .< μ], x[(x .> cut_low) .&& (x .< cut_high) .&& x .> μ])
    h = fit(Histogram, x, ifelse(left, cut_low, μ):bin_width:ifelse(left, μ, cut_high))
    n = length(x)

    # create fit functions
    f_fit(x, v) = pdf(truncated(Normal(v.μ, v.σ), ifelse(left, cut_low, μ), ifelse(left, μ, cut_high)), x)
    f_fit_n(x, v) = n * f_fit(x, v)
    
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μ = ConstValueDist(μ),
        σ = weibull_from_mx(ps.peak_sigma, 1.5*ps.peak_sigma)
    )

    # create fit model
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    
    v_init  = mean(pseudo_prior)

    f_loglike = let cut_low = ifelse(left, cut_low, μ), cut_high = ifelse(left, μ, cut_high),  x = x
        v -> (-1) * loglikelihood(truncated(Normal(v[1], v[2]), cut_low, cut_high), x)
    end

    # MLE
    opt_r = optimize(f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(opt_r.minimizer)

    if uncertainty
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance = inv(H)

        param_covariance = nothing
        if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)
        
        # TODO: p-values etc for unbinned fits
        # get p-value
        pval, chi2, dof = p_value(f_fit_n, h, v_ml)
        
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_fit_n, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) * x_unit for k in keys(v_ml)]...),
                  (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                  residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"

        result = merge(v_ml, )
    end

    # normalize nocut histogram to PDF of cut histogram
    h_pdf = Histogram(h_nocut.edges[1], h_nocut.weights ./ sum(h.weights) ./ step(h.edges[1]))

    report = (
        f_fit = t -> Base.Fix2(f_fit, v_ml)(t),
        h = h_pdf,
        μ = result.μ,
        σ = result.σ,
        gof = get(result, :gof, NamedTuple())
    )
    return (result = result, report = report)
end
export fit_half_centered_trunc_gauss



"""
    fit_half_trunc_gauss(x::Array, cuts::NamedTuple{(:low, :high, :max), Tuple{Float64, Float64, Float64}})
Fit a single truncated Gaussian to the data `x` between `cut.low` and `cut.high`. The peak center is fixed at `μ` and the peak is cut in half either in the left or right half.
# Returns `report` and `result` with:
    * `f_fit`: fitted function
    * `μ`: mean of the Gaussian
    * `σ`: standard deviation of the Gaussian
"""
function fit_half_trunc_gauss(x::Vector{<:Unitful.RealOrRealQuantity}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}; left::Bool=false, uncertainty::Bool=true) where T<:Unitful.RealOrRealQuantity
    @assert unit(cuts.low) == unit(cuts.high) == unit(cuts.max) == unit(x[1]) "Units of min_x, max_x and x must be the same"
    x_unit = unit(x[1])
    x, cut_low, cut_high, cut_max = ustrip.(x), ustrip(cuts.low), ustrip(cuts.high), ustrip(cuts.max)

    # get peak stats
    bin_width = get_friedman_diaconis_bin_width(x[(x .> cut_low) .&& (x .< cut_high)])
    x_min, x_max = minimum(x), maximum(x)
    h_nocut = fit(Histogram, x, x_min:bin_width:x_max)
    ps = estimate_single_peak_stats_th228(h_nocut)
    @debug "Peak stats: $ps"

    # cut peak out of data
    x = x[(x .> ifelse(left, cut_low, cut_max)) .&& (x .< ifelse(left, cut_max, cut_high))]
    h = fit(Histogram, x, ifelse(left, cut_low, cut_max):bin_width:ifelse(left, cut_max, cut_high))
    n = length(x)

    # create fit functions
    f_fit(x, v) = pdf(truncated(Normal(v.μ, v.σ), ifelse(left, cut_low, cut_max), ifelse(left, cut_max, cut_high)), x)
    f_fit_n(x, v) = n * f_fit(x, v)
    
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μ = Normal(ps.peak_pos, ps.peak_sigma/4),
        σ = weibull_from_mx(ps.peak_sigma, 1.5*ps.peak_sigma)
    )

    # create fit model
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    
    v_init  = mean(pseudo_prior)

    f_loglike = let cut_low = cut_low, cut_high = cut_high, cut_max = cut_max, left = left, x = x
        v -> (-1) * loglikelihood(truncated(Normal(v[1], v[2]), ifelse(left, cut_low, cut_max), ifelse(left, cut_max, cut_high)), x)
    end

    # fit data
    opt_r = optimize(f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(opt_r.minimizer)

    if uncertainty
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance = inv(H)

        param_covariance = nothing
        if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)
        
        # TODO: p-values etc for unbinned fits
        # get p-value
        pval, chi2, dof = p_value(f_fit_n, h, v_ml)
        
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_fit_n, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) * x_unit for k in keys(v_ml)]...),
                (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"

        result = merge(v_ml, )
    end

    # normalize nocut histogram to PDF of cut histogram
    h_pdf = Histogram(h_nocut.edges[1], h_nocut.weights ./ sum(h.weights) ./ step(h.edges[1]))

    report = (
        f_fit = t -> Base.Fix2(f_fit, v_ml)(t),
        h = h_pdf,
        μ = result.μ,
        σ = result.σ,
        gof = get(result, :gof, NamedTuple())
    )

    return (result = result, report = report)
end
export fit_half_trunc_gauss

#############
# Binned fits
#############

"""
    fit_binned_trunc_gauss(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real

Perform a binned fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_gauss` function consisting of a gaussian peak multiplied with an amplitude n.

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_binned_trunc_gauss(h_nocut::Histogram, cuts::NamedTuple{(:low, :high, :max), Tuple{<:T, <:T, <:T}}=(low = NaN, high = NaN, max = NaN); uncertainty::Bool=true) where T<:Unitful.RealOrRealQuantity
    # get cut windows
    @assert unit(cuts.low) == unit(cuts.high) == unit(cuts.max) "Units of min_x, max_x and x must be the same"
    cut_low, cut_high, cut_max = ustrip(cuts.low), ustrip(cuts.high), ustrip(cuts.max)
    x_min, x_max, bin_width = first(h_nocut.edges[1]), last(h_nocut.edges[1]), step(h_nocut.edges[1])
    cut_low, cut_high = ifelse(isnan(cut_low), x_min, cut_low), ifelse(isnan(cut_high), x_max, cut_high)


    # get peak stats
    ps = estimate_single_peak_stats_th228(h_nocut)
    @debug "Peak stats: $ps"

    # create cutted histogram
    h = h_nocut
    cut_idxs = collect(sort(findall(x -> x in Interval(cut_low, cut_high), h.edges[1])))
    if length(cut_idxs) != length(h.edges[1])
        weights = h.weights[cut_idxs]
        edges = if first(cut_idxs)-1 == 0
            h.edges[1][sort(push!(cut_idxs, last(cut_idxs)-1))]
        else
            h.edges[1][sort(push!(cut_idxs, first(cut_idxs)-1))]
        end
        h = Histogram(edges, weights)
    end

    # create fit function
    f_fit(x, v) = v.n * gauss_pdf(x, v.μ, v.σ) * heaviside(x - cut_low) * heaviside(cut_high - x)
    
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                μ = Normal(ps.peak_pos, ps.peak_sigma/4),
                σ = weibull_from_mx(ps.peak_sigma, 1.5*ps.peak_sigma),
                n = weibull_from_mx(ps.peak_counts, 2.0*ps.peak_counts), 
            )
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)
    
    # create loglikehood function
    f_loglike = let f_fit=f_fit, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    if uncertainty
        f_loglike_array(v) = - f_loglike(array_to_tuple(v, v_ml))

        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        param_covariance = nothing
        if !all(isfinite.(H)) || all(iszero.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        # get p-value
        pval, chi2, dof = p_value(f_fit, h, v_ml)
        
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_fit, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"

        result = merge(v_ml, )
    end


    report = (
            f_fit = x -> Base.Fix2(f_fit, v_ml)(x) * bin_width,
            h = h_nocut,
            μ = result.μ,
            σ = result.σ,
            gof = get(result, :gof, NamedTuple())
        )
    return (result = result, report = report)
end
export fit_binned_trunc_gauss


"""
    fit_binned_double_gauss(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real

Perform a binned fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_double_gauss` function consisting of a double gaussian peak.
The priors for the first gaussian peak are given by the `ps` tuple. For the priors of the second gaussian peak a wide window around the first peak is used.

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_binned_double_gauss(h::Histogram, ps::NamedTuple; uncertainty::Bool=true)

    # define double gaussina fit function
    f_double_gauss(x,v) = double_gaussian(x, v.μ1, v.σ1, v.n1, v.μ2, v.σ2, v.n2)

    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                #μ1 = Normal(ps.peak_pos, 5*ps.peak_sigma),
                μ1 = Uniform(ps.peak_pos-5*ps.peak_sigma, ps.peak_pos+5*ps.peak_sigma),
                #σ1 = Normal(ps.peak_sigma, 2*ps.peak_sigma),
                σ1 = Uniform(0.1*ps.peak_sigma, 5*ps.peak_sigma),
                n1 = Uniform(0.01*ps.peak_counts, 5*ps.peak_counts),
                #µ2 = Normal(ps.peak_pos, 5*ps.peak_sigma),
                µ2 = Uniform(0, 1200),
                #σ2 = Normal(ps.peak_sigma, 2*ps.peak_sigma),
                σ2 = Uniform(0.5*ps.peak_sigma, 5*ps.peak_sigma),
                n2 = Uniform(0.01*ps.peak_counts, 5*ps.peak_counts)
            )
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=f_double_gauss, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    if uncertainty
        f_loglike_array = let f_fit=double_gaussian, h=h
            v -> - hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v...) : 0, h)
        end

    
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        param_covariance = nothing
        if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        # get p-value 
        pval, chi2, dof = p_value(f_double_gauss, h, v_ml)
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_double_gauss, h, v_ml)

        @debug "Best Fit values"
        @debug "μ1: $(v_ml.μ1) ± $(v_ml_err.μ1)"
        @debug "σ1: $(v_ml.σ1) ± $(v_ml_err.σ1)"
        @debug "n1: $(v_ml.n1) ± $(v_ml_err.n)1"
        @debug "μ2: $(v_ml.μ2) ± $(v_ml_err.μ2)"
        @debug "σ2: $(v_ml.σ2) ± $(v_ml_err.σ2)"
        @debug "n2: $(v_ml.n2) ± $(v_ml_err.n2)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ1: $(v_ml.μ1)"
        @debug "σ1: $(v_ml.σ1)"
        @debug "n1: $(v_ml.n1)"
        @debug "μ2: $(v_ml.μ2)"
        @debug "σ2: $(v_ml.σ2)"
        @debug "n2: $(v_ml.n2)"

        result = merge(v_ml, )
    end
    report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(f_double_gauss, v_ml)(x),
            f_gauss_1 = x -> aoe_compton_signal_peakshape(x, v_ml.μ1, v_ml.σ1, v_ml.n1),
            f_gauss_2 = x -> aoe_compton_signal_peakshape(x, v_ml.μ2, v_ml.σ2, v_ml.n2)
        )
    return result, report
end
export fit_binned_double_gauss
