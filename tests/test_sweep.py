from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from experiments._sweep import (
    SearchSpace,
    TrialSeedOutcome,
    random_search,
    select_best_per_family,
    select_reference_trial_for_all_families,
    write_loss_curve_plots,
    write_training_params,
    write_tuning_log,
)


def test_search_space_samples_each_kind() -> None:
    space = SearchSpace(
        distributions={
            "lr": ("loguniform", 1e-3, 1e-1),
            "hidden": ("choice", [8, 16, 32]),
            "spread": ("uniform", 0.0, 1.0),
        }
    )
    rng = random.Random(0)

    sample = space.sample(rng)

    assert 1e-3 <= sample["lr"] <= 1e-1
    assert sample["hidden"] in {8, 16, 32}
    assert 0.0 <= sample["spread"] <= 1.0


def test_search_space_rejects_unknown_kind() -> None:
    space = SearchSpace(distributions={"x": ("normal", 0, 1)})

    with pytest.raises(ValueError):
        space.sample(random.Random(0))


def test_random_search_shares_trial_samples_across_families() -> None:
    space = SearchSpace(distributions={"lr": ("loguniform", 1e-3, 1e-1)})
    seen: dict[str, list[float]] = {"a": [], "b": []}

    def train_fn(family: str, hp: dict, seed: int) -> TrialSeedOutcome:
        seen[family].append(hp["lr"])
        return TrialSeedOutcome(
            val_accuracy=hp["lr"],
            test_accuracy=hp["lr"] + 0.01,
            train_seconds=0.1,
        )

    trials = random_search(
        families=["a", "b"],
        search_space=space,
        seeds=[0, 1],
        n_trials=3,
        sweep_seed=42,
        train_fn=train_fn,
    )

    assert len(trials) == 6
    assert seen["a"] == seen["b"]


def test_select_best_picks_highest_val_accuracy_per_family() -> None:
    space = SearchSpace(distributions={"x": ("choice", [0.1, 0.2, 0.3])})

    def train_fn(family: str, hp: dict, seed: int) -> TrialSeedOutcome:
        # Family "good" prefers x=0.3; family "bad" prefers x=0.1
        val = hp["x"] if family == "good" else (0.4 - hp["x"])
        return TrialSeedOutcome(
            val_accuracy=val, test_accuracy=val + 0.05, train_seconds=0.0
        )

    trials = random_search(
        families=["good", "bad"],
        search_space=space,
        seeds=[0],
        n_trials=10,
        sweep_seed=7,
        train_fn=train_fn,
    )
    selections = select_best_per_family(trials)

    by_family = {sel.family: sel for sel in selections}
    assert by_family["good"].selected_hyperparameters["x"] == 0.3
    assert by_family["bad"].selected_hyperparameters["x"] == 0.1


def test_select_reference_trial_uses_one_shared_trial_index() -> None:
    space = SearchSpace(distributions={"x": ("choice", [0.1, 0.2, 0.3])})

    def train_fn(family: str, hp: dict, seed: int) -> TrialSeedOutcome:
        val = hp["x"] if family == "complex" else (0.4 - hp["x"])
        return TrialSeedOutcome(
            val_accuracy=val,
            test_accuracy=val + 0.05,
            train_seconds=0.0,
        )

    trials = random_search(
        families=["complex", "real"],
        search_space=space,
        seeds=[0],
        n_trials=10,
        sweep_seed=7,
        train_fn=train_fn,
    )

    independent = select_best_per_family(trials)
    shared = select_reference_trial_for_all_families(trials)

    by_family_independent = {sel.family: sel for sel in independent}
    by_family_shared = {sel.family: sel for sel in shared}
    assert by_family_independent["complex"].selected_trial_index != (
        by_family_independent["real"].selected_trial_index
    )
    assert by_family_shared["complex"].selected_trial_index == (
        by_family_shared["real"].selected_trial_index
    )


def test_write_tuning_log_writes_markdown_and_json(tmp_path: Path) -> None:
    space = SearchSpace(distributions={"x": ("choice", [1, 2])})

    def train_fn(family: str, hp: dict, seed: int) -> TrialSeedOutcome:
        return TrialSeedOutcome(
            val_accuracy=0.5 + 0.1 * hp["x"],
            test_accuracy=0.5 + 0.1 * hp["x"] + 0.01,
            train_seconds=0.1,
        )

    trials = random_search(
        families=["c"],
        search_space=space,
        seeds=[0, 1],
        n_trials=2,
        sweep_seed=0,
        train_fn=train_fn,
    )
    selections = select_best_per_family(trials)

    write_tuning_log(
        tmp_path,
        task_name="test",
        sweep_config={"n_trials": 2, "seeds": [0, 1], "sweep_seed": 0},
        trials=trials,
        selections=selections,
    )

    payload = json.loads((tmp_path / "trials.json").read_text())
    assert len(payload["trials"]) == 2
    assert len(payload["selections"]) == 1
    assert (tmp_path / "tuning_log.md").exists()
    md = (tmp_path / "tuning_log.md").read_text()
    assert "Selected configuration per family" in md
    assert "All trials" in md


def test_random_search_checkpoints_and_resumes(tmp_path: Path) -> None:
    space = SearchSpace(distributions={"x": ("choice", [1, 2])})
    checkpoint_path = tmp_path / "checkpoint.json"
    calls: list[tuple[str, int, int]] = []

    def train_fn(family: str, hp: dict, seed: int) -> TrialSeedOutcome:
        calls.append((family, hp["x"], seed))
        return TrialSeedOutcome(
            val_accuracy=float(hp["x"]),
            test_accuracy=float(hp["x"]) + seed,
            train_seconds=0.1,
            extra={"train_loss_curve": [1.0, 0.5], "final_train_loss": 0.5},
        )

    first = random_search(
        families=["a", "b"],
        search_space=space,
        seeds=[0, 1],
        n_trials=2,
        sweep_seed=0,
        train_fn=train_fn,
        checkpoint_path=checkpoint_path,
        progress=False,
    )
    assert checkpoint_path.exists()
    assert len(calls) == 8

    def fail_if_called(family: str, hp: dict, seed: int) -> TrialSeedOutcome:
        raise AssertionError(f"resume should skip {family=} {hp=} {seed=}")

    resumed = random_search(
        families=["a", "b"],
        search_space=space,
        seeds=[0, 1],
        n_trials=2,
        sweep_seed=0,
        train_fn=fail_if_called,
        checkpoint_path=checkpoint_path,
        resume=True,
        progress=False,
    )

    assert [trial.to_dict() for trial in resumed] == [
        trial.to_dict() for trial in first
    ]


def test_training_params_and_loss_plots_are_written(tmp_path: Path) -> None:
    space = SearchSpace(distributions={"x": ("choice", [1])})

    def train_fn(family: str, hp: dict, seed: int) -> TrialSeedOutcome:
        return TrialSeedOutcome(
            val_accuracy=0.8,
            test_accuracy=0.7,
            train_seconds=0.1,
            extra={
                "parameter_count": 12,
                "train_loss_curve": [1.0, 0.8, 0.6],
                "final_train_loss": 0.6,
            },
        )

    trials = random_search(
        families=["complex", "real"],
        search_space=space,
        seeds=[0, 1],
        n_trials=1,
        sweep_seed=0,
        train_fn=train_fn,
        progress=False,
    )
    selections = select_reference_trial_for_all_families(trials)
    params_path = write_training_params(
        tmp_path,
        sweep_config={"n_trials": 1, "seeds": [0, 1], "sweep_seed": 0},
        trials=trials,
    )
    artifacts = write_loss_curve_plots(tmp_path, trials=trials, selections=selections)

    params = json.loads(params_path.read_text())
    assert len(params["runs"]) == 4
    assert params["runs"][0]["hyperparameters"] == {"x": 1}
    assert params["runs"][0]["train_loss_steps"] == 3
    assert (tmp_path / "loss_curves_all.png").exists()
    assert (tmp_path / "loss_curves_selected.png").exists()
    assert set(artifacts) == {"loss_curves_all", "loss_curves_selected"}
