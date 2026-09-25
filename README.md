# Neural Networks in Complex Spaces

Code and committed results for the paper. Everything needed to reproduce its
numbers, tables and figures is here; nothing else is.

- `cvnn/` — complex-valued layers, activations, baselines (including the
  $U(1)$-constrained real convolution) and manifest helpers.
- `experiments/` — the experiment drivers that produced `results/`.
- `scripts/` — the activation characterisation and the figure scripts.
- `results/` — one directory per reported configuration, each with a
  schema-conforming `manifest.json`.
- `docs/` — the baseline and tuning-budget rules the protocol follows, the
  RadioML setup guide, and the manifest schema.
- `tests/` — correctness tests, run in CI.

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
uv run pytest
```

`uv sync` resolves to a PyTorch build for a recent CUDA runtime, which falls
back to CPU on an older driver. To pin a build for a CUDA 12.x driver:

```bash
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
```

Each manifest's `environment` block records the build actually used.

## Data

The synthetic RF, quantum and EEG experiments generate their own data. RadioML
2018.01A is access-gated and not bundled; [`docs/radioml.md`](docs/radioml.md)
covers acquisition, the expected path, and the class-order sidecar.

## Reproducing the paper

Every command writes a manifest, raw runs and a summary, and accepts
`--resume`. Commands with `$ACT` were run once per activation in
`crelu zrelu modrelu cardioid siglog`.

| Paper result | Command | Output |
|---|---|---|
| Converged RF stress tests | `experiments/rf/representation_stress_tests.py --preset full --device cuda` | `results/rf_representation_stress_tests_full` |
| RF activation sweep | `experiments/rf/representation_stress_tests.py --preset full --device cuda --activation $ACT --tests qam_representation mixed_representation unit_power_mixed psk_representation` | `results/rf_activation_$ACT` |
| Quantum pilot | `experiments/physics/quantum_wavefunction.py --preset full --device cuda` | `results/physics_quantum_wavefunction_full` |
| Quantum activation sweep | `experiments/physics/quantum_wavefunction.py --preset full --device cuda --activation $ACT --tests potential_inverse global_phase_shift` | `results/physics_quantum_activation_$ACT` |
| EEG pilot | `experiments/neuro/eeg_analytic_signal.py --preset full --device cuda` | `results/neuro_eeg_analytic_signal_full` |
| EEG activation sweep | `experiments/neuro/eeg_analytic_signal.py --preset full --device cuda --activation $ACT --tests amplitude_event phase_amplitude_coupling` | `results/neuro_eeg_activation_$ACT` |
| EEG learning-rate sweep | `experiments/neuro/eeg_analytic_signal.py --preset standard --device cuda --learning-rate LR --tests amplitude_event phase_amplitude_coupling`, LR in `0.0003 0.001 0.003 0.01 0.03` | `results/neuro_eeg_lr_LR` |
| RadioML, 7 classes | `experiments/rf/sweep_radioml.py` with `$RADIOML_7` below, `--sample-length 128 --activation crelu` (and `zrelu`) | `results/radioml_geom7_{crelu,zrelu}` |
| RadioML, 10 classes | as above, adding `32QAM 128QAM 256QAM` to the modulations | `results/radioml_geom10_crelu` |
| RadioML, length 256 | as the 7-class run with `--sample-length 256` | `results/radioml_geom7_len256_crelu` |
| Clipping intervention | `experiments/rf/sweep_radioml.py --preset subset --activation crelu --n-trials 16 --seeds 0 1 2 3 4 5 --model-families $ALL` with and without `--grad-clip-norm 1.0` | `results/radioml_clip_intervention_{noclip,clip1.0}` |
| $U(1)$ ablation | as the 7-class run with `--model-families complex real_equivariant real_matched_params real_stacked real_polar real_magnitude` | `results/radioml_u1_ablation` |
| Activation gallery | `scripts/characterize_activations.py` | `results/activation_characterization` |
| Figures | `scripts/plot_crossover_figure.py`, `scripts/plot_paper_figures.py` | `paper/figures/` |

All commands are run as `uv run python <command>`. The RadioML arguments are:

```bash
RADIOML_7="--device cuda --modulations BPSK QPSK 8PSK 16QAM 64QAM 4ASK 16APSK \
  --snr-db-levels -14 -10 -6 -2 2 6 10 14 18 --max-per-class-per-snr 256 \
  --n-trials 16 --seeds 0 1 2 3 4 --model-families $ALL"
ALL="complex real_stacked real_matched_params real_matched_flops real_polar real_phase real_magnitude"
```

Add `--dataset-cache-dir <dir>` to reuse the filtered RadioML subset across
sweeps.

## Manifests and provenance

Each `manifest.json` conforms to
[`docs/result_manifest.schema.json`](docs/result_manifest.schema.json) and
records the git commit, a dirty-tree flag, Python / PyTorch / CUDA versions,
device and dtype, the full configuration, the seeds and the metrics.
Sweep-restart `checkpoint.json` files are excluded; they are bookkeeping, not
results.

The runs were executed from a working tree that was still dirty relative to
the commit their manifests name, so manifests carry that earlier hash with
`git_dirty: true`. The committed code is what should be used to reproduce
them.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
