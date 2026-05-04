"""Audit committed result artifacts for paper-track consistency."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RADIOML_ACTIVATION_DIRS: Mapping[str, str] = {
    "crelu": "radioml_modulation_sweep_crelu",
    "cardioid": "radioml_modulation_sweep_cardioid",
    "siglog": "radioml_modulation_sweep_siglog",
    "modrelu": "radioml_modulation_sweep_modrelu",
    "zrelu": "radioml_modulation_sweep_zrelu",
}

COMPARABLE_CONFIG_KEYS: tuple[str, ...] = (
    "architecture",
    "cache_data",
    "data_path",
    "dtype",
    "kernel_size",
    "max_per_class_per_snr",
    "model_families",
    "modulations",
    "n_trials",
    "real_activation",
    "sample_length",
    "search_space",
    "seeds",
    "snr_db_levels",
    "sweep_seed",
    "val_fraction",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"expected JSON object at {path}"
        raise TypeError(msg)
    return payload


def _actual_snr_keys(summary: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for selection in summary.get("matched_selections", []):
        if not isinstance(selection, dict):
            continue
        extra = selection.get("selected_extra", {})
        if not isinstance(extra, dict):
            continue
        per_seed = extra.get("test_accuracy_by_snr_db_per_seed", [])
        if not isinstance(per_seed, list):
            continue
        for item in per_seed:
            if isinstance(item, dict):
                keys.update(str(snr) for snr in item)
    return keys


def _complex_best_trial(summary: dict[str, Any]) -> int:
    trials_path = Path(str(summary["_trials_path"]))
    trials_payload = _load_json(trials_path)
    complex_trials = [
        trial
        for trial in trials_payload.get("trials", [])
        if isinstance(trial, dict) and trial.get("family") == "complex"
    ]
    if not complex_trials:
        msg = f"no complex trials in {trials_path}"
        raise ValueError(msg)
    best = max(complex_trials, key=lambda trial: float(trial["val_accuracy_mean"]))
    return int(best["trial_index"])


def audit_radioml_activations(*, strict_dirty: bool) -> list[str]:
    messages: list[str] = []
    errors: list[str] = []
    reference_config: dict[str, Any] | None = None

    for activation, directory in RADIOML_ACTIVATION_DIRS.items():
        result_dir = RESULTS / directory
        summary_path = result_dir / "summary.json"
        manifest_path = result_dir / "manifest.json"
        trials_path = result_dir / "trials.json"
        for path in (summary_path, manifest_path, trials_path):
            if not path.exists():
                errors.append(f"missing {path.relative_to(ROOT)}")
                continue
        if errors and not summary_path.exists():
            continue

        summary = _load_json(summary_path)
        summary["_trials_path"] = str(trials_path)
        manifest = _load_json(manifest_path)
        config = summary.get("config", {})
        if not isinstance(config, dict):
            errors.append(f"{summary_path.relative_to(ROOT)} has no config object")
            continue

        if config.get("activation") != activation:
            errors.append(
                f"{summary_path.relative_to(ROOT)} activation is "
                f"{config.get('activation')!r}, expected {activation!r}"
            )

        comparable = {key: config.get(key) for key in COMPARABLE_CONFIG_KEYS}
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            differing = [
                key
                for key in COMPARABLE_CONFIG_KEYS
                if comparable.get(key) != reference_config.get(key)
            ]
            errors.append(
                f"{summary_path.relative_to(ROOT)} differs from CReLU config on "
                f"{differing}"
            )

        expected_snr_keys = {str(int(snr)) for snr in config.get("snr_db_levels", [])}
        actual_snr_keys = _actual_snr_keys(summary)
        if expected_snr_keys != actual_snr_keys:
            errors.append(
                f"{summary_path.relative_to(ROOT)} per-SNR keys "
                f"{sorted(actual_snr_keys, key=int)} do not match config "
                f"{sorted(expected_snr_keys, key=int)}"
            )

        best_trial = _complex_best_trial(summary)
        matched = summary.get("matched_selections", [])
        if not isinstance(matched, list) or not matched:
            errors.append(f"{summary_path.relative_to(ROOT)} has no matched selections")
        else:
            selected_indices = {
                int(selection["selected_trial_index"])
                for selection in matched
                if isinstance(selection, dict)
            }
            if selected_indices != {best_trial}:
                errors.append(
                    f"{summary_path.relative_to(ROOT)} matched trial indices "
                    f"{sorted(selected_indices)} do not equal complex best "
                    f"trial {best_trial}"
                )

        environment = manifest.get("environment", {})
        dirty = isinstance(environment, dict) and environment.get("git_dirty") is True
        if dirty:
            message = (
                f"dirty manifest: {manifest_path.relative_to(ROOT)} "
                f"(commit {environment.get('git_commit')})"
            )
            if strict_dirty:
                errors.append(message)
            else:
                messages.append("WARN " + message)

        messages.append(
            f"OK {directory}: activation={activation}, "
            f"snrs={sorted(actual_snr_keys, key=int)}, best_trial={best_trial}"
        )

    legacy = RESULTS / "radioml_modulation_sweep" / "summary.json"
    if legacy.exists():
        legacy_summary = _load_json(legacy)
        legacy_keys = _actual_snr_keys(legacy_summary)
        legacy_config = legacy_summary.get("config", {})
        if isinstance(legacy_config, dict):
            legacy_expected = {
                str(int(snr)) for snr in legacy_config.get("snr_db_levels", [])
            }
            if legacy_keys != legacy_expected:
                messages.append(
                    "WARN legacy radioml_modulation_sweep has mismatched SNR "
                    f"keys {sorted(legacy_keys, key=int)} vs config "
                    f"{sorted(legacy_expected, key=int)}; paper figures use "
                    "radioml_modulation_sweep_crelu instead"
                )

    if errors:
        joined = "\n".join(errors)
        msg = f"result audit failed:\n{joined}"
        raise RuntimeError(msg)
    return messages


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-dirty",
        action="store_true",
        help="fail if any audited manifest records dirty code state",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    for message in audit_radioml_activations(strict_dirty=args.strict_dirty):
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
