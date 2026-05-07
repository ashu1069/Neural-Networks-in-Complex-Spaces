# RadioML 2018.01A

The synthetic RF modulation benchmark in `experiments/rf/synthetic_modulation.py`
is a stand-in. The real-data version uses **RadioML 2018.01A**, the standard
public benchmark for over-the-air-style modulation classification.

## Acquisition

1. Register at [DeepSig Datasets](https://www.deepsig.ai/datasets) and accept
   the licence. The archive is gated; we cannot bundle it here.
2. Download `2018.01.OSC.0001_1024.hdf5` (~22 GB compressed,
   `GOLD_XYZ_OSC.0001_1024.hdf5` after extraction in some mirrors).
3. Verify the SHA-256 against DeepSig's published checksum if available.

## Path convention

By default the loader looks for the file at:

```text
data/GOLD_XYZ_OSC.0001_1024.hdf5
```

Override with the `path` argument or the `--data-path` CLI flag where
applicable. For Colab or any machine where the archive lives outside the repo,
copy the example paths config and edit it once:

```bash
cp config/radioml_paths.example.json config/radioml_paths.json
```

Example `config/radioml_paths.json` for a mounted Google Drive:

```json
{
  "radioml_2018_01a": {
    "data_path": "/content/drive/MyDrive/datasets/RadioML/GOLD_XYZ_OSC.0001_1024.hdf5",
    "classes_path": "/content/drive/MyDrive/datasets/RadioML/classes-fixed.json"
  }
}
```

`config/radioml_paths.json` is git-ignored. The sweep and telemetry scripts use
this resolution order: explicit `--data-path` / `--classes-path`, then
`RADIOML_DATA_PATH` / `RADIOML_CLASSES_PATH`, then `config/radioml_paths.json`,
then the default `data/` path. If you keep the config somewhere else, pass
`--paths-config /path/to/radioml_paths.json` or set `RADIOML_PATHS_CONFIG`.

**Do not commit the dataset file.** The repo's `.gitignore` excludes `data/`.

Keep the fixed class sidecar next to the HDF5 when possible:

```text
data/classes-fixed.json
```

The original `classes.txt` distributed with the dataset is known to list the
24 modulations in the wrong one-hot order. The loader therefore prefers
`classes-fixed.json` or `classes-fixed.txt` next to the HDF5, then falls back to
an HDF5-embedded class list or the canonical paper order.

## File layout

The archive contains three top-level HDF5 datasets:

| name | shape | dtype | meaning |
|---|---|---|---|
| `X` | `(N, 1024, 2)` | `float32` | IQ sequence; last dim is `(I, Q)` |
| `Y` | `(N, 24)` | `int8`/`float32` | one-hot modulation label |
| `Z` | `(N, 1)` | `int8` | SNR in dB |

`N ≈ 2,555,904` examples. The 24 modulations and 26 SNR levels
(-20 to +30 in 2 dB steps) are listed in
[`experiments/rf/radioml.py`](../experiments/rf/radioml.py) as
`RADIOML_2018_01A_MODULATIONS` and `RADIOML_2018_01A_SNR_DB`.

## Loader

```python
from experiments.rf.radioml import load_radioml_2018_01a

# Full archive (heavy — minutes to load, GBs of memory)
data = load_radioml_2018_01a("data/GOLD_XYZ_OSC.0001_1024.hdf5")

# Filtered subset for fast iteration
data = load_radioml_2018_01a(
    "data/GOLD_XYZ_OSC.0001_1024.hdf5",
    modulations=["BPSK", "QPSK", "8PSK"],
    snr_db_levels=[-10, -5, 0, 5, 10, 15, 20],
    max_per_class_per_snr=256,
    sample_length=128,  # trim from 1024 down to first 128 samples
)
```

Common aliases such as `PSK8`, `QAM16`, `APSK16`, `ASK4`, and AM labels with
underscores are accepted and normalized to the fixed RadioML names (`8PSK`,
`16QAM`, `16APSK`, `4ASK`, `AM-SSB-WC`, etc.).

The returned object is the same `RFModulationData` dataclass the synthetic
benchmark produces, so the four-family scaffolding in
`experiments/rf/synthetic_modulation.py` and the sweep harness
`experiments/rf/sweep_radioml.py` operate on it unchanged.

## What we replicate vs. don't

The synthetic stand-in covers the *shape* of the task (per-sample IQ
classification, per-SNR breakdown, label imbalance handled by stratified
sampling). What it deliberately omits:

- **Pulse shaping.** Real RadioML symbols are root-raised-cosine shaped at
  8 samples per symbol; the synthetic version uses raw symbols.
- **Carrier frequency offset.** The synthetic data has zero CFO; RadioML
  has random CFO drawn per example.
- **Channel effects.** RadioML adds simulated multipath/Rician fading
  ([detailed in the dataset paper](https://arxiv.org/abs/1712.04578)); the
  synthetic data has only AWGN.

So the synthetic benchmark answers "do complex-valued networks have an
inductive bias for IQ-shape data," and RadioML answers "and does that
advantage survive the distortions you actually encounter."

## Cost notes

- **Loading** the full archive into memory takes ~3 minutes on an SSD and
  consumes ~25 GB. The loader's filtering args let you avoid this.
- **GPU sweeps** with the full 24-class / 26-SNR / 1024-sample setup at
  the upper end of the search space are multi-hour runs even on an A100.
  Start with a subset and a tractable sample length; scale up once the
  harness reproduces the sub-task numbers.
- **Caching** is enabled for capped sweep subsets so repeated trial/family
  runs reuse the same seed/filter split instead of re-scanning the HDF5 labels.
  Uncapped full-archive runs disable this automatically; use `--no-cache-data`
  if a capped subset is still too large for memory.

## Sweep presets

The `--preset` flag on `experiments/rf/sweep_radioml.py` selects a coherent
default bundle (modulations + SNR levels + sample length + search space).
Individual flags still override the preset.

### `--preset subset` (default)

3 PSK modulations (BPSK / QPSK / 8PSK), 8 even SNR levels
(-10, -6, -2, 2, 6, 10, 14, 18 dB), `max_per_class_per_snr=256`,
`sample_length=128`. Search space: hidden ∈ {16, 32, 64} (conv) /
{32, 64, 128} (mlp), batch ∈ {128, 256, 512}, steps ∈ {200, 400, 800}.
This is the regime the §3.4 RadioML headline used; runs in ~20-30 min on
A100. Output goes to `results/radioml_modulation_sweep_<activation>/`.

### `--preset full`

All 24 modulations, all 26 even SNR levels (-20 to +30 in 2 dB steps),
`max_per_class_per_snr=256` (≈160k samples after filtering),
`sample_length=1024` (RadioML's native length). Search space: hidden ∈
{32, 64, 128, 256} (conv), batch ∈ {256, 512, 1024}, steps ∈
{400, 800, 1600}. This brings the sweep into the RadioML literature's
evaluation regime. **Expect 5-15 hours on a single A100** for the default
16 trials × 3 seeds × 4 families = 192 runs, depending on which trial
samples land at the upper end of the search space. Output goes to
`results/radioml_modulation_sweep_full_<activation>/` to avoid clobbering
subset snapshots.

To halve cost: pass `--max-per-class-per-snr 128` (cuts samples 2×),
`--snr-db-levels -10 -6 -2 2 6 10 14 18` (cuts SNRs ~3×), or
`--seeds 0 1` (cuts seeds 1.5×).

### Recommended GPU command for the full scale-up

```bash
# Pull latest, confirm clean tree (CI dirty-manifest guard warns either way)
git pull && git status

# Full 24-class run on the headline activation (CReLU)
uv run python experiments/rf/sweep_radioml.py --device cuda \
    --preset full --activation crelu --seeds 0 1 2 --resume

# Optional: ablate a stable activation (zrelu) on the full task to test
# whether the robustness asymmetry survives at scale
uv run python experiments/rf/sweep_radioml.py --device cuda \
    --preset full --activation zrelu --seeds 0 1 2 --resume
```

When done:

```bash
grep '"git_dirty"' results/radioml_modulation_sweep_full_*/manifest.json
git add results/radioml_modulation_sweep_full_*/
git commit -m "Phase 5+: RadioML full-archive sweep (24 classes, 26 SNRs, 1024 samples)"
git push
```

Long sweeps write a seed-level checkpoint after every completed training run:

```text
results/radioml_modulation_sweep_full_<activation>/checkpoint.json
```

If Colab disconnects, rerun the same command with `--resume`; completed
`(family, trial, seed)` rows are skipped. Completed sweeps also write:

```text
training_params.json
loss_curves_all.png
loss_curves_selected.png
```

`training_params.json` is the flat run list with every family/trial/seed,
hyperparameter sample, metric, runtime, parameter count, and final training
loss. The PNGs plot all trial mean loss curves and the selected-trial seed
curves.

## Citation

If you publish RadioML results, cite:

> O'Shea, T. J., Roy, T., & Clancy, T. C. (2018). Over-the-air deep
> learning based radio signal classification. *IEEE Journal of Selected
> Topics in Signal Processing*, 12(1), 168-179.
