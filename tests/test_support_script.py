from __future__ import annotations

import json
import subprocess
import sys


def test_support_script_runs_on_cpu() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_torch_complex_support.py",
            "--device",
            "cpu",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["metadata"]["torch"]
    assert all(result["status"] == "pass" for result in payload["results"])
