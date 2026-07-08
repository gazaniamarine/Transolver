# NS turbulence comparison figures

Output of `sciml_showcase/ns_turbulence/compare_models.py`, run at 128x128
resolution on the real `nsforcing_*` dataset:

* `loss_curves_comparison.png` — Transolver vs. FNO train/test Rel-L2 over 300 epochs.
* `energy_spectrum_comparison.png` — 1D radial energy spectrum vs. Kolmogorov k^(-5/3) decay.
* `flow_comparison_showcase_{5,12,28}.png` — side-by-side vorticity field predictions and error maps for 3 test samples.
* `summary_metrics.md` — quantitative comparison table (test Rel-L2, parameter count, inference latency).

These are referenced from [`sciml_showcase/README.md`](../../sciml_showcase/README.md)
and the [root README](../../README.md).
