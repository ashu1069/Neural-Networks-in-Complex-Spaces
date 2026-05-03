"""Random-search sweep harness for benchmark experiments.

Implements the budget rule from `docs/tuning_budget.md`: every model family
gets the same number of trials (default 16) drawn from a single per-task
search distribution. For each trial, train across `seeds` random seeds and
record the per-seed metrics. Selection picks the trial with the highest
mean validation metric across seeds; the test metric for that trial's seeds
is what gets reported.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JsonValue = str | int | float | bool | None | dict[str, Any] | list[Any]
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class SearchSpace:
    """A flat search distribution over hyperparameters.

    Each entry maps a parameter name to either:
      - a `("uniform", low, high)` triple for a continuous uniform draw,
      - a `("loguniform", low, high)` triple for a log-uniform continuous draw,
      - a `("choice", [v1, v2, ...])` pair for a discrete choice.

    The same `SearchSpace` is shared across all model families in a sweep so
    no family gets a structural advantage from a wider search.
    """

    distributions: Mapping[str, tuple[Any, ...]]

    def sample(self, rng: random.Random) -> dict[str, Any]:
        sampled: dict[str, Any] = {}
        for name, spec in self.distributions.items():
            kind = spec[0]
            if kind == "uniform":
                _, low, high = spec
                sampled[name] = rng.uniform(low, high)
            elif kind == "loguniform":
                _, low, high = spec
                sampled[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
            elif kind == "choice":
                _, options = spec
                sampled[name] = rng.choice(options)
            else:
                msg = f"unsupported distribution kind: {kind}"
                raise ValueError(msg)
        return sampled


@dataclass(frozen=True)
class TrialResult:
    """Per-seed metrics for one (family, hyperparameter sample) pair."""

    family: str
    trial_index: int
    hyperparameters: dict[str, Any]
    seeds: list[int]
    val_accuracy_per_seed: list[float]
    test_accuracy_per_seed: list[float]
    train_seconds_per_seed: list[float]
    extra: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def val_accuracy_mean(self) -> float:
        return sum(self.val_accuracy_per_seed) / len(self.val_accuracy_per_seed)

    @property
    def test_accuracy_mean(self) -> float:
        return sum(self.test_accuracy_per_seed) / len(self.test_accuracy_per_seed)

    def to_dict(self) -> JsonObject:
        return {
            "family": self.family,
            "trial_index": self.trial_index,
            "hyperparameters": self.hyperparameters,
            "seeds": list(self.seeds),
            "val_accuracy_per_seed": list(self.val_accuracy_per_seed),
            "test_accuracy_per_seed": list(self.test_accuracy_per_seed),
            "train_seconds_per_seed": list(self.train_seconds_per_seed),
            "val_accuracy_mean": self.val_accuracy_mean,
            "test_accuracy_mean": self.test_accuracy_mean,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class FamilySelection:
    """The selected trial for a family after a sweep."""

    family: str
    selected_trial_index: int
    selected_val_accuracy_mean: float
    selected_test_accuracy_mean: float
    selected_test_accuracy_std: float
    selected_hyperparameters: dict[str, Any]
    selected_extra: dict[str, JsonValue]

    def to_dict(self) -> JsonObject:
        return {
            "family": self.family,
            "selected_trial_index": self.selected_trial_index,
            "selected_val_accuracy_mean": self.selected_val_accuracy_mean,
            "selected_test_accuracy_mean": self.selected_test_accuracy_mean,
            "selected_test_accuracy_std": self.selected_test_accuracy_std,
            "selected_hyperparameters": self.selected_hyperparameters,
            "selected_extra": dict(self.selected_extra),
        }


TrialFn = Callable[[str, dict[str, Any], int], "TrialSeedOutcome"]


@dataclass(frozen=True)
class TrialSeedOutcome:
    """Output of one (family, hyperparameter sample, seed) run."""

    val_accuracy: float
    test_accuracy: float
    train_seconds: float
    extra: JsonObject = field(default_factory=dict)


def random_search(
    *,
    families: Sequence[str],
    search_space: SearchSpace,
    seeds: Sequence[int],
    n_trials: int,
    sweep_seed: int,
    train_fn: TrialFn,
) -> list[TrialResult]:
    """Run a shared-budget random search across all families.

    The same hyperparameter samples are drawn once and reused across families,
    so every family sees the *same* trials - no family can luck into a
    better-suited search distribution.
    """

    if n_trials <= 0:
        msg = "n_trials must be positive"
        raise ValueError(msg)
    if not seeds:
        msg = "seeds must be non-empty"
        raise ValueError(msg)
    if not families:
        msg = "families must be non-empty"
        raise ValueError(msg)

    rng = random.Random(sweep_seed)
    sampled_trials: list[dict[str, Any]] = [
        search_space.sample(rng) for _ in range(n_trials)
    ]

    trials: list[TrialResult] = []
    for family in families:
        for trial_index, hyperparameters in enumerate(sampled_trials):
            val_per_seed: list[float] = []
            test_per_seed: list[float] = []
            seconds_per_seed: list[float] = []
            extra_per_seed: list[JsonObject] = []
            for seed in seeds:
                outcome = train_fn(family, hyperparameters, seed)
                val_per_seed.append(outcome.val_accuracy)
                test_per_seed.append(outcome.test_accuracy)
                seconds_per_seed.append(outcome.train_seconds)
                extra_per_seed.append(outcome.extra)
            aggregated_extra: JsonObject = (
                _aggregate_extras(extra_per_seed) if extra_per_seed else {}
            )
            trials.append(
                TrialResult(
                    family=family,
                    trial_index=trial_index,
                    hyperparameters=hyperparameters,
                    seeds=list(seeds),
                    val_accuracy_per_seed=val_per_seed,
                    test_accuracy_per_seed=test_per_seed,
                    train_seconds_per_seed=seconds_per_seed,
                    extra=aggregated_extra,
                )
            )
    return trials


def select_best_per_family(
    trials: Sequence[TrialResult],
) -> list[FamilySelection]:
    """For each family, pick the trial with the highest mean val accuracy."""

    by_family: dict[str, list[TrialResult]] = {}
    for trial in trials:
        by_family.setdefault(trial.family, []).append(trial)

    selections: list[FamilySelection] = []
    for family, family_trials in by_family.items():
        best = max(family_trials, key=lambda trial: trial.val_accuracy_mean)
        test_mean = best.test_accuracy_mean
        if len(best.test_accuracy_per_seed) >= 2:
            mean = test_mean
            variance = sum(
                (value - mean) ** 2 for value in best.test_accuracy_per_seed
            ) / (len(best.test_accuracy_per_seed) - 1)
            test_std = math.sqrt(variance)
        else:
            test_std = 0.0
        selections.append(
            FamilySelection(
                family=family,
                selected_trial_index=best.trial_index,
                selected_val_accuracy_mean=best.val_accuracy_mean,
                selected_test_accuracy_mean=test_mean,
                selected_test_accuracy_std=test_std,
                selected_hyperparameters=dict(best.hyperparameters),
                selected_extra=dict(best.extra),
            )
        )
    return selections


def write_tuning_log(
    output_dir: Path,
    *,
    task_name: str,
    sweep_config: JsonObject,
    trials: Sequence[TrialResult],
    selections: Sequence[FamilySelection],
) -> None:
    """Write `trials.json` and `tuning_log.md` for a completed sweep."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_payload: JsonObject = {
        "config": sweep_config,
        "trials": [trial.to_dict() for trial in trials],
        "selections": [selection.to_dict() for selection in selections],
    }
    (output_dir / "trials.json").write_text(
        json.dumps(trials_payload, indent=2, sort_keys=True) + "\n"
    )

    md_lines = [
        f"# Tuning Log - {task_name}",
        "",
        (
            "Random-search sweep following `docs/tuning_budget.md`: shared "
            "trial samples across all families, seeded per trial, selection "
            "by mean validation accuracy."
        ),
        "",
        f"- Trials per family: `{sweep_config.get('n_trials')}`",
        f"- Seeds per trial: `{sweep_config.get('seeds')}`",
        f"- Sweep seed: `{sweep_config.get('sweep_seed')}`",
        "- Search space: see `trials.json`",
        "",
        "## Selected configuration per family",
        "",
        ("| family | trial | val acc | test acc | test std | hyperparameters |"),
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for selection in selections:
        hp_str = ", ".join(
            f"{key}={_format_value(value)}"
            for key, value in sorted(selection.selected_hyperparameters.items())
        )
        md_lines.append(
            " | ".join(
                [
                    f"| `{selection.family}`",
                    str(selection.selected_trial_index),
                    f"{selection.selected_val_accuracy_mean:.4f}",
                    f"{selection.selected_test_accuracy_mean:.4f}",
                    f"{selection.selected_test_accuracy_std:.4f}",
                    f"{hp_str} |",
                ]
            )
        )
    md_lines.extend(
        [
            "",
            "## All trials (mean across seeds)",
            "",
            "| family | trial | val acc | test acc | hyperparameters |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for trial in trials:
        hp_str = ", ".join(
            f"{key}={_format_value(value)}"
            for key, value in sorted(trial.hyperparameters.items())
        )
        md_lines.append(
            " | ".join(
                [
                    f"| `{trial.family}`",
                    str(trial.trial_index),
                    f"{trial.val_accuracy_mean:.4f}",
                    f"{trial.test_accuracy_mean:.4f}",
                    f"{hp_str} |",
                ]
            )
        )
    (output_dir / "tuning_log.md").write_text("\n".join(md_lines) + "\n")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _aggregate_extras(extras: Sequence[JsonObject]) -> JsonObject:
    """Merge per-seed extras into per-key lists; numeric keys averaged too."""

    if not extras:
        return {}
    keys: set[str] = set()
    for extra in extras:
        keys.update(extra.keys())
    aggregated: JsonObject = {}
    for key in sorted(keys):
        values = [extra.get(key) for extra in extras]
        if all(
            isinstance(value, int | float) and not isinstance(value, bool)
            for value in values
        ):
            numeric_values = [float(value) for value in values]  # type: ignore[arg-type]
            aggregated[key + "_mean"] = sum(numeric_values) / len(numeric_values)
            aggregated[key + "_per_seed"] = numeric_values
        else:
            aggregated[key + "_per_seed"] = list(values)
    return aggregated


__all__ = [
    "FamilySelection",
    "JsonObject",
    "JsonValue",
    "SearchSpace",
    "TrialFn",
    "TrialResult",
    "TrialSeedOutcome",
    "random_search",
    "select_best_per_family",
    "write_tuning_log",
]
