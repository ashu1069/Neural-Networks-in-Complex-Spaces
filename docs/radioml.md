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
data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5
```

Override with the `path` argument or the `--data-path` CLI flag where
applicable. **Do not commit the file.** The repo's `.gitignore` excludes
`data/`.

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
data = load_radioml_2018_01a("data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5")

# Filtered subset for fast iteration
data = load_radioml_2018_01a(
    "data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5",
    modulations=["BPSK", "QPSK", "8PSK"],
    snr_db_levels=[-10, -5, 0, 5, 10, 15, 20],
    max_per_class_per_snr=256,
    sample_length=128,  # trim from 1024 down to first 128 samples
)
```

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

## Citation

If you publish RadioML results, cite:

> O'Shea, T. J., Roy, T., & Clancy, T. C. (2018). Over-the-air deep
> learning based radio signal classification. *IEEE Journal of Selected
> Topics in Signal Processing*, 12(1), 168-179.
