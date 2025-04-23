import subprocess
import sys
import os

def test_bench_cli_runs_without_error():
    env = os.environ.copy()
    env["MECH_QUICK"] = "1"
    ret = subprocess.run(
        [sys.executable, "-m", "mechanistic_interventions.evaluation.benchmark"],
        capture_output=True,
        env=env,
        timeout=5,
    )
    assert ret.returncode == 0
    assert ret.stdout.strip() == b"[]"
