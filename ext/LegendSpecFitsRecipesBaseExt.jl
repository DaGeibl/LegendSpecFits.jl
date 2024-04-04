# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsRecipesBaseExt

using RecipesBase
using Unitful, Formatting, Measurements, LaTeXStrings
using Measurements: value, uncertainty
using StatsBase, LinearAlgebra

function round_wo_units(x::Unitful.RealOrRealQuantity, digits::Integer=2)
    if unit(x) == NoUnits
        round(x, digits=digits)
    else
        round(unit(x), x, digits=2)
    end
end

# @recipe function f(x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where T<:Unitful.RealOrRealQuantity
@recipe function f(report::NamedTuple{(:f_fit, :μ, :σ, :n)}, x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where {T <: Unitful.RealOrRealQuantity}
    ylabel := "Normalized Counts"
    legend := :bottomright
    @series begin
        seriestype := :histogram
        bins --> :fd
        normalize --> :pdf
        label := "Data"
        ustrip(x[x .> cuts.low .&& x .< cuts.high])
    end
    @series begin
        color := :red
        label := "Normal Fit (μ = $(round_wo_units(report.μ, digits=2)), σ = $(round_wo_units(report.σ, digits=2)))"
        lw := 3
        ustrip(cuts.low):ustrip(Measurements.value(report.σ / 1000)):ustrip(cuts.high), t -> report.f_fit(t)
    end
end

@recipe function f(report:: NamedTuple{(:rt, :min_enc, :enc_grid_rt, :enc)})
    xlabel := "Rise Time ($(unit(first(report.rt))))"
    ylabel := "ENC (ADC)"
    grid := :true
    gridcolor := :black
    gridalpha := 0.2
    gridlinewidth := 0.5
    # xscale := :log10
    # yscale := :log10
    # xlims := (5e0, 2e1)
    @series begin
        seriestype := :scatter
        label := "ENC"
        report.enc_grid_rt*NoUnits, report.enc
    end
    @series begin
        seriestype := :hline
        label := "Min. ENC Noise (RT: $(report.rt))"
        color := :red
        linewidth := 2.5
        [report.min_enc]
    end
end

@recipe function f(report:: NamedTuple{(:ft, :min_fwhm, :e_grid_ft, :fwhm)})
    xlabel := "Flat-Top Time $(unit(first(report.ft)))"
    ylabel := "FWHM FEP"
    grid := :true
    gridcolor := :black
    gridalpha := 0.2
    gridlinewidth := 0.5
    # xscale := :log10
    # yscale := :log10
    ylims := (1, 8)
    xlims := (0.5, 5)
    @series begin
        seriestype := :scatter
        label := "FWHM"
        report.e_grid_ft*NoUnits, report.fwhm
    end
    @series begin
        seriestype := :hline
        label := "Min. FWHM $(round(u"keV", report.min_fwhm, digits=2)) (FT: $(report.ft))"
        color := :red
        linewidth := 2.5
        [report.min_fwhm]
    end
end

@recipe function f(report:: NamedTuple{(:wl, :min_sf, :a_grid_wl_sg, :sfs)})
    xlabel := "Window Length ($(unit(first(report.a_grid_wl_sg))))"
    ylabel := "SEP Surrival Fraction ($(unit(first(report.sfs))))"
    grid := :true
    gridcolor := :black
    gridalpha := 0.2
    gridlinewidth := 0.5
    # ylims := (0, 30)
    @series begin
        seriestype := :scatter
        label := "SF"
        ustrip.(report.a_grid_wl_sg), ustrip.(report.sfs)
    end
    @series begin
        seriestype := :hline
        label := "Min. SF $(report.min_sf) (WT: $(report.wl))"
        color := :red
        linewidth := 2.5
        [ustrip(Measurements.value(report.min_sf))]
    end
    @series begin
        seriestype := :hspan
        label := ""
        color := :red
        alpha := 0.1
        ustrip.([Measurements.value(report.min_sf)-Measurements.uncertainty(report.min_sf), Measurements.value(report.min_sf)+Measurements.uncertainty(report.min_sf)])
    end
end

@recipe function f(report::NamedTuple{(:v, :h, :f_fit, :f_sig, :f_lowEtail, :f_bck, :gof)}; show_label=true, show_fit=true, f_fit_x_step_scaling=1/100, _subplot=1)
    f_fit_x_step = ustrip(value(report.v.σ)) * f_fit_x_step_scaling
    legend := :topright
    ylim_max = max(3*value(report.f_sig(report.v.μ)), 3*maximum(report.h.weights))
    ylim_max = ifelse(ylim_max == 0.0, 1e5, ylim_max)
    ylim_min = 0.1*minimum(filter(x -> x > 0, report.h.weights))
    framestyle --> :box
    @series begin
        seriestype := :stepbins
        label := ifelse(show_label, "Data", "")
        bins --> :sqrt
        yscale := :log10
        ylims := (ylim_min, ylim_max)
        xlabel := "Energy (keV)"
        xlims := (minimum(report.h.edges[1]), maximum(report.h.edges[1]))
        ylabel := "Counts / $(round(step(report.h.edges[1]), digits=2)) keV"
        subplot --> _subplot
        LinearAlgebra.normalize(report.h, mode = :density)
    end
    if show_fit
        @series begin
            seriestype := :line
            if !isempty(report.gof)
                label := ifelse(show_label, "Best Fit (p = $(round(report.gof.pvalue, digits=2)))", "")
            else
                label := ifelse(show_label, "Best Fit", "")
            end
            color := :red
            subplot --> _subplot
            # ribbon := uncertainty.(report.f_fit.(minimum(report.h.edges[1]):0.1:maximum(report.h.edges[1])))
            minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), value.(report.f_fit.(minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1])))
        end
        @series begin
            seriestype := :line
            label := ifelse(show_label, "Signal", "")
            subplot --> _subplot
            color := :green
            minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_sig
        end
        @series begin
            seriestype := :line
            label := ifelse(show_label, "Low Energy Tail", "")
            subplot --> _subplot
            color := :blue
            minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_lowEtail
        end
        @series begin
            seriestype := :line
            label := ifelse(show_label, "Background", "")
            subplot --> _subplot
            margins --> (0, :mm)
            bottom_margin --> (-4, :mm)
            xlabel := ""
            xticks --> ([])
            ylabel := "Counts / $(round(step(report.h.edges[1]), digits=2)) keV"
            ylims := (ylim_min, ylim_max)
            xlims := (minimum(report.h.edges[1]), maximum(report.h.edges[1]))
            color := :black
            minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_bck
        end
        if !isempty(report.gof)
            ylims_res_max, ylims_res_min = 5, -5
            if any(report.gof.residuals_norm .> 5) || any(report.gof.residuals_norm .< -5)
                abs_max = 1.2*maximum(abs.(report.gof.residuals_norm))
                ylims_res_max, ylims_res_min = abs_max, -abs_max
            end
            layout --> @layout([a{0.8h}; b{0.2h}])
            margins --> (0, :mm)
            link --> :x
            @series begin
                seriestype := :hline
                ribbon := 3
                subplot --> _subplot + 1
                fillalpha := 0.5
                label := ""
                fillcolor := :lightgrey
                linecolor := :darkgrey
                [0.0]
            end
            @series begin
                seriestype := :hline
                ribbon := 1
                subplot --> _subplot + 1
                fillalpha := 0.5
                label := ""
                fillcolor := :grey
                linecolor := :darkgrey
                [0.0]
            end
            @series begin
                seriestype := :scatter
                subplot --> _subplot + 1
                label := ""
                title := ""
                markercolor --> :black
                ylabel --> "Residuals (σ)"
                xlabel := "Energy (keV)"
                link --> :x
                top_margin --> (0, :mm)
                ylims --> (ylims_res_min, ylims_res_max)
                xlims := (minimum(report.h.edges[1]), maximum(report.h.edges[1]))
                yscale --> :identity
                if ylims_res_max == 5
                    yticks --> ([-3, 0, 3])
                end
                report.gof.bin_centers, report.gof.residuals_norm
            end
        else
            @series begin
                seriestype := :line
                label := ifelse(show_label, "Background", "")
                subplot --> _subplot
                margins --> (0, :mm)
                bottom_margin --> (-4, :mm)
                xlabel := "Energy (keV)"
                xticks --> ([])
                ylabel := "Counts / $(round(step(report.h.edges[1]), digits=2)) keV"
                ylims := (ylim_min, ylim_max)
                xlims := (minimum(report.h.edges[1]), maximum(report.h.edges[1]))
                color := :black
                minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_bck
            end
        end
    end
end

@recipe function f(report::NamedTuple{((:v, :h, :f_fit, :f_sig, :f_bck))}; f_fit_x_step_scaling=1/100)
    f_fit_x_step = ustrip(value(report.v.σ)) * f_fit_x_step_scaling
    xlabel := "A/E (a.u.)"
    ylabel := "Counts"
    legend := :topleft
    ylims := (1, max(1.5*report.f_sig(report.v.μ), 1.5*maximum(report.h.weights)))
    @series begin
        seriestype := :stepbins
        label := "Data"
        bins --> :sqrt
        LinearAlgebra.normalize(report.h, mode = :density)
    end
    @series begin
        seriestype := :line
        label := "Best Fit"
        color := :red
        minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_fit
    end
    @series begin
        seriestype := :line
        label := "Signal"
        color := :green
        minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_sig
    end
    @series begin
        seriestype := :line
        label := "Background"
        color := :black
        minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_bck
    end
end

@recipe function f(x::Vector{T}, cut::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where T<:Unitful.RealOrRealQuantity
    ylabel := "Counts"
    legend := :topright
    xlims := (median(x) - std(x), median(x) + std(x))
    @series begin
        seriestype := :stephist
        label := "Data"
        bins --> :sqrt
        x
    end
    @series begin
        seriestype := :vline
        label := ""
        color := :red
        linewidth := 1.5
        [cut.low, cut.high]
    end
    @series begin
        seriestype := :vspan
        label := "Cut Window"
        color := :red
        alpha := 0.1
        fillrange := cut.high
        [cut.low, cut.high]
    end
end

@recipe function f(report::NamedTuple{(:h_calsimple, :h_uncal, :c, :fep_guess, :peakhists, :peakstats)}; cal=true)
    ylabel := "Counts"
    legend := :topright
    yscale := :log10
    if cal
        h = LinearAlgebra.normalize(report.h_calsimple, mode = :density)
        xlabel := "Energy (keV)"
        xlims := (0, 3000)
        xticks := (0:200:3000, ["$i" for i in 0:200:3000])
        ylims := (0.2, maximum(h.weights)*1.1)
        fep_guess = 2614.5
    else
        h = LinearAlgebra.normalize(report.h_uncal, mode = :density)
        xlabel := "Energy (ADC)"
        xlims := (0, 1.2*report.fep_guess)
        xticks := (0:3000:1.2*report.fep_guess, ["$i" for i in 0:3000:1.2*report.fep_guess])
        ylims := (0.2, maximum(h.weights)*1.1)
        fep_guess = report.fep_guess
    end
    @series begin
        seriestype := :stepbins
        label := "Energy"
        h
    end
    y_vline = 0.2:1:maximum(h.weights)*1.1
    @series begin
        seriestype := :line
        label := "FEP Guess"
        color := :red
        linewidth := 1.5
        fill(fep_guess, length(y_vline)), y_vline
    end
end

@recipe function f(report_ctc::NamedTuple{(:peak, :window, :fct, :bin_width, :bin_width_qdrift, :e_peak, :e_ctc, :qdrift_peak, :h_before, :h_after, :fwhm_before, :fwhm_after, :report_before, :report_after)})
    layout := (1, 3)
    thickness_scaling := 1.0
    xtickfontsize := 12
    xlabelfontsize := 14
    ylabelfontsize := 14
    ytickfontsize := 12
    legendfontsize := 10
    size := (1000, 300)
    margin := (8, :mm)
    @series begin
        seriestype := :histogram2d
        bins := (ustrip.(unit(first(report_ctc.e_peak)), minimum(report_ctc.e_peak):report_ctc.bin_width:maximum(report_ctc.e_peak)), quantile(report_ctc.qdrift_peak, 0.01):report_ctc.bin_width_qdrift:quantile(report_ctc.qdrift_peak, 0.99))
        color := :inferno
        xlabel := "Energy"
        ylabel := "QDrift"
        title := "Before Correction"
        titlelocation := (0.5, 1.1)
        xlims := (2600, 2630)
        ylims := (0, quantile(report_ctc.qdrift_peak, 0.99))
        yformatter := :plain
        legend := :none
        colorbar_scale := :log10
        subplot := 1
        report_ctc.e_peak, report_ctc.qdrift_peak
    end
    @series begin
        seriestype := :histogram2d
        bins := (ustrip.(unit(first(report_ctc.e_peak)), minimum(report_ctc.e_peak):report_ctc.bin_width:maximum(report_ctc.e_peak)), quantile(report_ctc.qdrift_peak, 0.01):report_ctc.bin_width_qdrift:quantile(report_ctc.qdrift_peak, 0.99))
        color := :magma
        xlabel := "Energy"
        ylabel := "QDrift"
        title := "After Correction"
        xlims := (2600, 2630)
        titlelocation := (0.5, 1.1)
        xlims := (2600, 2630)
        ylims := (0, quantile(report_ctc.qdrift_peak, 0.99))
        yformatter := :plain
        legend := :none
        colorbar_scale := :log10
        subplot := 3
        report_ctc.e_ctc, report_ctc.qdrift_peak
    end
    @series begin
        seriestype := :stepbins
        color := :red
        label := "Before CTC"
        xlabel := "Energy (keV)"
        ylabel := "Counts"
        yscale := :log10
        subplot := 2
        report_ctc.h_before
    end
    @series begin
        seriestype := :stepbins
        color := :green
        label := "After CTC"
        xlabel := "Energy (keV)"
        ylabel := "Counts"
        title := "FWHM $(round(u"keV", report_ctc.fwhm_after, digits=2))"
        titlelocation := (0.5, 1.1)
        xlims := (2600, 2630)
        xticks := (2600:10:2630)
        legend := :bottomright
        yscale := :log10
        subplot := 2
        report_ctc.h_after
    end
end

@recipe function f(report_window_cut::NamedTuple{(:h, :f_fit, :x_fit, :low_cut, :high_cut, :low_cut_fit, :high_cut_fit, :center, :σ)})
    xlims := (value(ustrip(report_window_cut.center - 5*report_window_cut.σ)), value(ustrip(report_window_cut.center + 5*report_window_cut.σ)))
    @series begin
        report_window_cut.h        
    end
    @series begin
        seriestype := :line
        label := "Best Fit"
        color := :red
        linewidth := 3
        report_window_cut.x_fit, report_window_cut.f_fit
    end
    @series begin
        seriestype := :vline
        label := "Cut Window"
        color := :green
        linewidth := 3
        ustrip.([report_window_cut.low_cut, report_window_cut.high_cut])
    end
    @series begin
        seriestype := :vline
        label := "Fit Window"
        color := :orange
        linewidth := 3
        ustrip.([report_window_cut.low_cut_fit, report_window_cut.high_cut_fit])
    end
    @series begin
        seriestype := :vspan
        label := ""
        color := :lightgreen
        alpha := 0.2
        ustrip.([report_window_cut.low_cut, report_window_cut.high_cut])
    end

end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof)}; plot_ribbon=true, xerrscaling=1, additional_pts=NamedTuple())
    thickness_scaling := 2.0
    xlims := (0, 1.2*value(maximum(report.x)))
    framestyle := :box
    xformatter := :plain
    yformatter := :plain
    layout --> @layout([a{0.8h}; b{0.2h}])
    margins --> (-11.5, :mm)
    link --> :x
    @series begin
        seriestype := :line
        subplot := 1
        xticks --> :none
        if !isempty(report.gof)
            label := "Best Fit (p = $(round(report.gof.pvalue, digits=2)))"
        else
            label := "Best Fit"
        end
        color := :orange
        linewidth := 2
        fillalpha := 0.2
        if plot_ribbon
            ribbon := uncertainty.(report.f_fit.(0:1:1.2*value(maximum(report.x))))
        end
        0:1:1.2*value(maximum(report.x)), value.(report.f_fit.(0:1:1.2*value(maximum(report.x))))
    end
    @series begin
        seriestype := :scatter
        subplot := 1
        if xerrscaling == 1
            label := "Data"
        else
            label := "Data (Error x$(xerrscaling))"
        end
        markercolor --> :black
        value.(report.x), report.y
    end
    if !isempty(additional_pts)
        @series begin
            seriestype := :scatter
            subplot --> 1
            if xerrscaling == 1
                label := "Data not used for fit"
            else
                label := "Data not used for fit (Error x$(xerrscaling))"
            end
            ms --> 3
            markershape --> :circle
            markerstrokecolor --> :black
            linewidth --> 0.5
            markercolor --> :silver
            xerror := uncertainty.(additional_pts.x) .* xerrscaling
            value.(additional_pts.x), additional_pts.y
        end
    end
    @series begin
        seriestype := :hline
        ribbon --> 3
        subplot --> 2
        fillalpha --> 0.5
        label --> ""
        fillcolor --> :lightgrey
        linecolor --> :darkgrey
        [0.0]
    end
    @series begin
        seriestype --> :hline
        ribbon --> 1
        subplot --> 2
        fillalpha --> 0.5
        label --> ""
        fillcolor --> :grey
        linecolor --> :darkgrey
        [0.0]
    end
    if !isempty(additional_pts)
        @series begin
            seriestype := :scatter
            label --> :none
            ms --> 3
            markershape --> :circle
            markerstrokecolor --> :black
            linewidth --> 0.5
            markercolor --> :silver
            subplot --> 2
            value.(additional_pts.x), additional_pts.residuals_norm
        end
    end
    @series begin
        seriestype --> :scatter
        subplot --> 2
        label --> ""
        markercolor --> :black
        ylabel --> "Residuals (σ)"
        ylims --> (-5, 5)
        yticks --> ([-3, 0, 3])
        value.(report.x), report.gof.residuals_norm
    end
end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof, :e_unit, :qbb, :type)}; additional_pts=NamedTuple())
    bottom_margin --> (0, :mm)
    if report.type == :fwhm
        y_max = value(maximum(report.y))
        additional_pts = if !isempty(additional_pts)
            fwhm_cal = report.f_fit.(ustrip.(additional_pts.peaks)) .* report.e_unit
            y_max = max(y_max, value(maximum(ustrip.(report.e_unit, additional_pts.fwhm))))
            (x = additional_pts.peaks, y = additional_pts.fwhm,
                residuals_norm = (value.(fwhm_cal) .- additional_pts.fwhm)./ uncertainty.(fwhm_cal))
        else
            NamedTuple()
        end
        xlabel := "Energy (keV)"
        legend := :bottomright
        framestyle := :box
        xlims := (0, 3000)
        xticks := (0:500:3000, ["$i" for i in 0:500:3000])
        @series begin
            grid --> :all
            xerrscaling --> 1
            additional_pts --> additional_pts
            (par = report.par, f_fit = report.f_fit, x = report.x, y = report.y, gof = report.gof)
        end
        @series begin
            seriestype := :hline
            label := L"Q_{\beta \beta}:" * " $(round(u"keV", report.qbb, digits=2))"
            color := :green
            fillalpha := 0.2
            linewidth := 2.5
            xticks := :none
            ylabel := "FWHM"
            ylims := (0, 1.2*y_max)
            subplot := 1
            ribbon := uncertainty(report.qbb)
            [value(report.qbb)]
        end
    end
end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof, :e_unit, :type)}; xerrscaling=1, additional_pts=NamedTuple())
    bottom_margin --> (0, :mm)
    if report.type == :cal
        additional_pts = if !isempty(additional_pts)
            μ_cal = report.f_fit.(additional_pts.μ) .* report.e_unit
            (x = additional_pts.μ, y = additional_pts.peaks, 
                residuals_norm = (value.(μ_cal) .- additional_pts.peaks)./ uncertainty.(μ_cal))
        else
            NamedTuple()
        end
        xlabel := "Energy (ADC)"
        legend := :bottomright
        framestyle := :box
        xlims := (0, 21000)
        xticks := (0:2000:22000)
        @series begin
            grid --> :all
            xerrscaling := xerrscaling
            additional_pts := additional_pts
            (par = report.par, f_fit = report.f_fit, x = report.x, y = report.y, gof = report.gof)
        end
        @series begin
            seriestype := :hline
            label := L"Q_{\beta \beta}"
            color := :green
            fillalpha := 0.2
            linewidth := 2.5
            xticks := :none
            ylabel := "Energy ($(report.e_unit))"
            ylims := (0, 1.2*value(maximum(report.y)))
            yticks := (500:500:3000)
            subplot := 1
            [2039]
        end
    end
end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof, :e_unit, :label_y, :label_fit)})
    layout --> @layout([a{0.8h}; b{0.2h}])
    margins --> (0, :mm)
    link --> :x
    size := (1200, 700)
    layout --> @layout([a{0.7h}; b{0.3h}])
    xmin = floor(Int, minimum(report.x)/100)*100
    xmax = ceil(Int, maximum(report.x)/100)*100
    @series begin
        seriestype := :line
        subplot --> 1
        color := :orange
        ms := 3
        linewidth := 3
        label := report.label_fit
        ribbon := uncertainty.(report.f_fit(report.x))
        report.x, value.(report.f_fit(report.x))
    end
    @series begin
            ylabel := "$(report.label_y) (a.u.)"
            seriestype := :scatter
            subplot --> 1
            color := :black 
            ms := 3
            yguidefontsize := 18
            xguidefontsize := 18
            ytickfontsize := 12
            xtickfontsize := 12
            legendfontsize := 14
            foreground_color_legend := :silver
            background_color_legend := :white
            xlims := (xmin,xmax)
            xticks := (xmin:250:xmax, fill(" ", length(xmin:250:xmax) ))
            ylims := (0.98 * (Measurements.value(minimum(report.y)) - Measurements.uncertainty(median(report.y))), 1.02 * (Measurements.value(maximum(report.y)) + Measurements.uncertainty(median(report.y)) ) )
            margin := (10, :mm)
            bottom_margin := (-7, :mm)
            framestyle := :box
            grid := :false
            label := "Compton band fits: Gaussian $(report.label_y)(A/E)"
            report.x, report.y
    end

    @series begin
        seriestype := :hline
        ribbon := 3
        subplot --> 2
        fillalpha := 0.5
        label := ""
        fillcolor := :lightgrey
        linecolor := :darkgrey
        [0.0]
    end
    @series begin
        xlabel := "Energy ($(report.e_unit))"
        ylabel := "Residuals (σ) \n"
        seriestype := :scatter
        subplot --> 2
        color := :black 
        ms := 3
        label := false
        framestyle := :box
        grid := :false
        xlims := (xmin,xmax)
        xticks := xmin:250:xmax
        ylims := (floor(minimum(report.gof.residuals_norm)-1), ceil(maximum(report.gof.residuals_norm))+1)
        yticks := [-5,0,5]
        yguidefontsize := 18
        xguidefontsize := 18
        ytickfontsize := 12
        xtickfontsize := 12
        report.x, report.gof.residuals_norm
    end
end

end # module LegendSpecFitsRecipesBaseExt