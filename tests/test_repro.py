from __future__ import annotations

import json
from pathlib import Path

from cvnn.repro import Environment, ResultManifest


def test_result_manifest_writes_json(tmp_path: Path) -> None:
    manifest = ResultManifest(
        schema_version="0.1.0",
        run_id="smoke-test",
        created_at="2026-01-01T00:00:00+00:00",
        environment=Environment(
            python="3.12.0",
            platform="test-platform",
            macos=None,
            torch="2.0.0",
            device="cpu",
            dtype="complex64",
            git_commit="abc123",
        ),
        config={"experiment": "smoke"},
        seeds=[0],
        metrics={"loss": 0.0},
        dataset={"name": "synthetic"},
        artifacts={"log": "outputs/smoke.log"},
        notes="unit test fixture",
    )

    path = tmp_path / "manifest.json"
    manifest.write_json(path)

    data = json.loads(path.read_text())
    assert data["schema_version"] == "0.1.0"
    assert data["environment"]["device"] == "cpu"
    assert data["metrics"]["loss"] == 0.0
