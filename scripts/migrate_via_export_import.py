"""
Migrate local MLflow runs to DagsHub using the mlflow-export-import tool.

Two phases:
    1) EXPORT: dump local mlruns/ to a staging directory (fast, local-only).
    2) IMPORT: upload the staging directory to DagsHub (slow, network-bound).

Each phase streams the underlying subprocess stdout/stderr line-by-line with
timestamps so you can watch what's happening in real time.

Usage:
    python scripts/migrate_via_export_import.py              # both phases
    python scripts/migrate_via_export_import.py export       # export only
    python scripts/migrate_via_export_import.py import       # import only
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import dagshub

REPO_ROOT = Path(__file__).parent.parent
LOCAL_MLRUNS = REPO_ROOT / "mlruns"
EXPORT_DIR = REPO_ROOT / "mlruns_export"
DAGSHUB_OWNER = "Alex44lel"
DAGSHUB_REPO = "eeg_biomarkers_models"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def stream(cmd: list[str], env: dict, label: str) -> int:
    log(f"$ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{ts()}] {label} | {line.rstrip()}", flush=True)
    rc = proc.wait()
    log(f"{label} finished rc={rc} in {time.time() - t0:.1f}s")
    return rc


def do_export() -> int:
    if not LOCAL_MLRUNS.exists():
        log(f"ERROR: {LOCAL_MLRUNS} does not exist")
        return 1
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = f"file://{LOCAL_MLRUNS}"
    env["PYTHONUNBUFFERED"] = "1"
    env["LOG_LEVEL"] = env.get("LOG_LEVEL", "INFO")

    log(f"EXPORT src={env['MLFLOW_TRACKING_URI']}  dst={EXPORT_DIR}")
    cmd = [
        sys.executable, "-m", "mlflow_export_import.bulk.export_experiments",
        "--experiments", "all",
        "--output-dir", str(EXPORT_DIR),
    ]
    return stream(cmd, env, "EXPORT")


def do_import() -> int:
    if not EXPORT_DIR.exists() or not any(EXPORT_DIR.iterdir()):
        log(f"ERROR: {EXPORT_DIR} is empty. Run the export phase first.")
        return 1

    log(f"Connecting to DagsHub {DAGSHUB_OWNER}/{DAGSHUB_REPO}...")
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["LOG_LEVEL"] = env.get("LOG_LEVEL", "INFO")
    log(f"IMPORT dst MLFLOW_TRACKING_URI={env.get('MLFLOW_TRACKING_URI')}")

    cmd = [
        sys.executable, "-m", "mlflow_export_import.bulk.import_experiments",
        "--input-dir", str(EXPORT_DIR),
        "--use-threads", "True",
    ]
    return stream(cmd, env, "IMPORT")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in {"all", "export", "import"}:
        log(f"Unknown mode '{mode}'. Use: all | export | import")
        return 2

    t0 = time.time()
    if mode in ("export", "all"):
        rc = do_export()
        if rc != 0:
            return rc
    if mode in ("import", "all"):
        rc = do_import()
        if rc != 0:
            return rc
    log(f"Total wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
