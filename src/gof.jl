# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
"""
    `p_value(f_fit, h, v_ml)` : calculate p-value of chi2 test for goodness-of-fit
 input:
 * f_fit --> function handle of fit function (peakshape)
 * h --> histogram of data
 * v_ml --> best-fit parameters
 output:
 * pval --> p-value of chi2 test
 * chi2 --> chi2 value
 * dof --> degrees of freedom
"""

function prepare_data(h::Histogram{<:Real,1})
    # get bin center, width and counts from histogrammed data
    bin_edges = first(h.edges)
    counts = h.weights
    bin_centers = (bin_edges[begin:end-1] .+ bin_edges[begin+1:end]) ./ 2
    bin_widths = bin_edges[begin+1:end] .- bin_edges[begin:end-1]
    return counts, bin_widths, bin_centers
end
export prepare_data


function get_model_counts(f_fit::Base.Callable,v_ml::NamedTuple,bin_centers::StepRangeLen,bin_widths::StepRangeLen)
    model_func  = Base.Fix2(f_fit, v_ml) # fix the fit parameters to ML best-estimate
    model_counts = bin_widths.*map(energy->model_func(energy), bin_centers) # evaluate model at bin center (= binned measured energies)
    return model_counts
end
export get_model_counts


""" baseline: p-value via least-squares """
function p_value(f_fit::Base.Callable, h::Histogram{<:Real,1},v_ml::NamedTuple)
    # prepare data
    counts, bin_widths, bin_centers = prepare_data(h)

    # get peakshape of best-fit 
    model_counts = get_model_counts(f_fit, v_ml, bin_centers,bin_widths)
    
    # calculate chi2
    chi2    = sum((model_counts[model_counts.>0]-counts[model_counts.>0]).^2 ./ model_counts[model_counts.>0])
    npar    = length(v_ml)
    dof    = length(counts[model_counts.>0])-npar
    pval    = ccdf(Chisq(dof),chi2)
    if any(model_counts.<=5)
              @warn "WARNING: bin with <=$(round(minimum(model_counts),digits=0)) counts -  chi2 test might be not valid"
    else  
         @debug "p-value = $(round(pval,digits=2))"
    end
    return pval, chi2, dof
end
export p_value

""" alternative p-value via loglikelihood ratio"""
function p_value_LogLikeRatio(f_fit::Base.Callable, h::Histogram{<:Real,1},v_ml::NamedTuple)
    # prepare data
    counts, bin_widths, bin_centers = prepare_data(h)

    # get peakshape of best-fit 
    model_counts =get_model_counts(f_fit, v_ml, bin_centers,bin_widths)
    
    # calculate chi2
    chi2    = sum((model_counts[model_counts.>0]-counts[model_counts.>0]).^2 ./ model_counts[model_counts.>0])
    npar    = length(v_ml)
    dof    = length(counts[model_counts.>0])-npar
    pval    = ccdf(Chisq(dof),chi2)
    if any(model_counts.<=5)
              @warn "WARNING: bin with <=$(minimum(model_counts)) counts -  chi2 test might be not valid"
    else  
         @debug "p-value = $(round(pval,digits=2))"
    end
    chi2   = 2*sum(model_counts.*log.(model_counts./counts)+model_counts-counts)
    pval   = ccdf(Chisq(dof),chi2)
return pval, chi2, dof
end
export p_value_LogLikeRatio

""" alternative p-value via Monte Carlo. Warning: can be computational expensive!"""
function p_value_MC(f_fit::Base.Callable, h::Histogram{<:Real,1},ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background)},v_ml::NamedTuple,;n_samples::Int64=1000)
    counts, bin_widths, bin_centers = prepare_data(h) # get data 
   
    # get peakshape of best-fit and maximum likelihood value
    model_func  = Base.Fix2(f_fit, v_ml) # fix the fit parameters to ML best-estimate
    model_counts = bin_widths.*map(energy->model_func(energy), bin_centers) # evaluate model at bin center (= binned measured energies)
    loglike_bf = -hist_loglike(model_func,h) 

    # draw sample for each bin
    dists = Poisson.(model_counts) # create poisson distribution for each bin
    counts_mc_vec = rand.(dists,n_samples) # randomized histogram counts
    counts_mc = [ [] for _ in 1:n_samples ] #re-structure data_samples_vec to array of arrays, there is probably a better way to do this...
    for i = 1:n_samples
        counts_mc[i] = map(x -> x[i],counts_mc_vec)
    end
    
    # fit every sample histogram and calculate max. loglikelihood
    loglike_bf_mc = NaN.*ones(n_samples)
    h_mc = h # make copy of data histogram
    for i=1:n_samples
        h_mc.weights = counts_mc[i] # overwrite counts with MC values
        result_fit_mc, report = fit_single_peak_th228(h_mc, ps ; uncertainty=false) # fit MC histogram
        fit_par_mc   = result_fit_mc[(:μ, :σ, :n, :step_amplitude, :skew_fraction, :skew_width, :background)]
        model_func_sample  = Base.Fix2(f_fit, fit_par_mc) # fix the fit parameters to ML best-estimate
        loglike_bf_mc[i] = -hist_loglike(model_func_sample,h_mc) # loglikelihood for best-fit
    end

    # calculate p-value
    pval= sum(loglike_bf_mc.<=loglike_bf)./n_samples
    return pval 
end
export p_value_MC