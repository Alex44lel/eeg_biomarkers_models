"""
Migrate all local MLflow runs to DagsHub.

Copies params, metrics, tags, and artifacts for every run in every experiment.
Skips runs already present on DagsHub (matched by mlflow.runName + source run_id tag).
"""

import os
import tempfile
import shutil
from pathlib import Path

import dagshub
import mlflow
from mlflow.tracking import MlflowClient

LOCAL_URI = str(Path(__file__).parent.parent / "mlruns")
DAGSHUB_OWNER = "Alex44lel"
DAGSHUB_REPO = "eeg_biomarkers_models"

SKIP_TAGS = {"mlflow.log-model.history"}  # large / not useful remotely


def migrate():
    local = MlflowClient(LOCAL_URI)

    print("Connecting to DagsHub...")
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    remote = MlflowClient()

    # Build set of already-migrated source run IDs
    migrated = set()
    for exp in remote.search_experiments():
        for run in remote.search_runs(exp.experiment_id):
            src = run.data.tags.get("migrated_from_run_id")
            if src:
                migrated.add(src)

    experiments = [e for e in local.search_experiments() if e.name != "Default"]
    print(f"Found {len(experiments)} experiments to migrate.\n")

    for exp in experiments:
        # Get or create experiment on remote
        remote_exp = remote.get_experiment_by_name(exp.name)
        if remote_exp is None:
            remote_exp_id = remote.create_experiment(exp.name)
        else:
            remote_exp_id = remote_exp.experiment_id

        runs = local.search_runs(exp.experiment_id, order_by=["start_time ASC"])
        # Separate parent and child runs so parents are created first
        parents = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")]
        children = [r for r in runs if r.data.tags.get("mlflow.parentRunId")]

        print(f"[{exp.name}] {len(runs)} runs ({len(parents)} parent, {len(children)} child)")

        # Map local run_id -> remote run_id for parent linking
        id_map: dict[str, str] = {}

        for run in parents + children:
            _migrate_run(local, remote, remote_exp_id, run, id_map, migrated)

    print("\nMigration complete.")


def _migrate_run(local, remote, remote_exp_id, run, id_map, migrated):
    src_id = run.info.run_id
    run_name = run.data.tags.get("mlflow.runName", src_id[:8])

    if src_id in migrated:
        print(f"  SKIP {run_name} (already migrated)")
        id_map[src_id] = None  # unknown remote id but already exists
        return

    # Resolve parent run id
    parent_local_id = run.data.tags.get("mlflow.parentRunId")
    parent_remote_id = id_map.get(parent_local_id) if parent_local_id else None

    new_run = remote.create_run(
        experiment_id=remote_exp_id,
        run_name=run_name,
        start_time=run.info.start_time,
        tags={"migrated_from_run_id": src_id},
    )
    new_id = new_run.info.run_id
    id_map[src_id] = new_id

    # Params
    params = run.data.params
    for i in range(0, len(params), 100):
        chunk = dict(list(params.items())[i:i+100])
        remote.log_batch(new_id, params=[mlflow.entities.Param(k, v) for k, v in chunk.items()])

    # Metrics (all steps)
    for key in run.data.metrics:
        history = local.get_metric_history(src_id, key)
        for i in range(0, len(history), 1000):
            chunk = history[i:i+1000]
            remote.log_batch(new_id, metrics=[
                mlflow.entities.Metric(m.key, m.value, m.timestamp, m.step) for m in chunk
            ])

    # Tags (skip internal/large ones)
    tags = {k: v for k, v in run.data.tags.items() if k not in SKIP_TAGS}
    if parent_remote_id:
        tags["mlflow.parentRunId"] = parent_remote_id
    for k, v in tags.items():
        remote.set_tag(new_id, k, v)

    # Artifacts
    artifacts = local.list_artifacts(src_id)
    if artifacts:
        local_artifact_uri = local.get_run(src_id).info.artifact_uri
        local_artifact_path = local_artifact_uri.replace("file://", "")
        if os.path.exists(local_artifact_path):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_copy = os.path.join(tmp, "artifacts")
                shutil.copytree(local_artifact_path, tmp_copy)
                remote.log_artifacts(new_id, tmp_copy)

    # Set final status
    remote.update_run(new_id, status=run.info.status)
    print(f"  OK   {run_name}")


if __name__ == "__main__":
    migrate()
