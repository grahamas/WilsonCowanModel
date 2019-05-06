plotspecs = [
  # SpaceTimePlot(),
  SubsampledPlot(
    plot_type=WaveStatsPlot,
    time_subsampler=Subsampler(
      Δ = 0.05,
      window = (1.0, 1.6)
    ),
    space_subsampler=Subsampler(
        window = (5.0,Inf)
      ),
    peak_interpolation_n=2
    )
]