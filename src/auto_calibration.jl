# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


struct EnergyCalibrationDiagnostics{H<:Histogram{<:Integer,1}}
    cal_hist::H
end


"""
    autocal_energy(E_raw::AbstractArray{<:Real})

Compute an energy calibration from raw reconstructed energy deposition values.
"""
function autocal_energy(E_raw::AbstractArray{<:Real})
    cal_hist_binning = 0:0.5:3000
    quantile_perc = 0.995
    calib_constant = 2614.5 / quantile(E_raw, quantile_perc)
    f_calib = Base.Fix1(*, calib_constant * u"keV")
    E_cal_keV = ustrip.(f_calib.(E_raw))
    cal_hist = fit(Histogram, E_cal_keV, cal_hist_binning)
    return (result = f_calib, diagnostics = EnergyCalibrationDiagnostics(cal_hist))
end

export autocal_energy


"""
    calibrate_energy!(e::AbstractArray{<:Real}, pars::PropDict)

Calibrate energy values in-place.
"""
function calibrate_energy!(e::Array{T}, pars::PropDict) where T<:Real
    e .*= pars.m_calib
    e .+= pars.n_calib
end

