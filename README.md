# Transolver: Reproduction & Turbulence/Aerodynamics Benchmarking

This repository builds on **Transolver** (Wu et al., *ICML 2024*,
[thuml/Transolver](https://github.com/thuml/Transolver)). The core
architecture ([`Physics_Attention.py`](Physics_Attention.py)) and the
standard-benchmark code
([`PDE-Solving-StandardBenchmark/`](PDE-Solving-StandardBenchmark),
[`Car-Design-ShapeNetCar/`](Car-Design-ShapeNetCar),
[`Airfoil-Design-AirfRANS/`](Airfoil-Design-AirfRANS)) are the original
authors' work, reproduced here. Everything under [`sciml_showcase/`](sciml_showcase)
is my own extension on top of that codebase (see "My contributions" below).

Transolver computes attention among a small number of learned physical
states rather than directly over mesh points, which gives it mesh-independent,
geometry-general behavior on PDEs discretized over complex, irregular grids.

<p align="center">
<img src="pic/Transolver.png" height="250" alt="Transolver architecture overview" align="center" />
<br><br>
<b>Figure 1.</b> Overview of Transolver (from Wu et al., 2024).
</p>

## Repository layout

* [`Physics_Attention.py`](Physics_Attention.py), [`PDE-Solving-StandardBenchmark/`](PDE-Solving-StandardBenchmark) — upstream Transolver architecture and the six standard PDE benchmarks (Darcy, NS, elasticity, plasticity, pipe, airfoil).
* [`Car-Design-ShapeNetCar/`](Car-Design-ShapeNetCar), [`Airfoil-Design-AirfRANS/`](Airfoil-Design-AirfRANS) — upstream industrial design benchmarks.
* [`sciml_showcase/`](sciml_showcase) — my extensions: a Transolver-vs-FNO turbulence benchmark and an unstructured-grid airfoil RANS pipeline.

## My contributions

* **[`sciml_showcase/ns_turbulence`](sciml_showcase/ns_turbulence)** — Transolver vs. FNO benchmark on 2D Navier-Stokes vorticity dynamics (128x128 resolution), including energy-spectrum validation against the Kolmogorov k^(-5/3) inertial-range decay.
* **[`sciml_showcase/airfoil_rans`](sciml_showcase/airfoil_rans)** — unstructured-grid (VTU/VTP, PyVista) training pipeline for airfoil RANS surrogates, validated end-to-end on synthetic geometries.

## Results

On the 2D Navier-Stokes turbulence benchmark
([`sciml_showcase/ns_turbulence`](sciml_showcase/ns_turbulence), 128x128
resolution, 300 epochs each, default hyperparameters):

**FNO: 0.144 rel-L2 at 2.80M params vs. Transolver: 0.395 rel-L2 at 1.42M
params — FNO is ~2.7x more accurate and ~3.8x faster at inference;
Transolver uses ~2x fewer parameters.** See
[`sciml_showcase/README.md`](sciml_showcase/README.md) for the full
comparison table and a caveat on an earlier degenerate 64x64 dataset.

![Loss curves](results/comparison_ns/loss_curves_comparison.png)
![Energy spectrum](results/comparison_ns/energy_spectrum_comparison.png)

See [`sciml_showcase/README.md`](sciml_showcase/README.md) for the full
writeup, including the airfoil RANS pipeline status.

## Citation (upstream work)

If you use Transolver itself, please cite the original paper:

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

Paper: [arXiv:2402.02366](https://arxiv.org/abs/2402.02366) ·
Upstream repository: [thuml/Transolver](https://github.com/thuml/Transolver)

## Acknowledgement (upstream)

The upstream authors acknowledge the following repositories for valuable
code and datasets:

* https://github.com/neuraloperator/neuraloperator
* https://github.com/neuraloperator/Geo-FNO
* https://github.com/thuml/Latent-Spectral-Models
* https://github.com/Extrality/AirfRANS

## License

MIT License, Copyright (c) 2024 THUML @ Tsinghua University. See [LICENSE](LICENSE).
