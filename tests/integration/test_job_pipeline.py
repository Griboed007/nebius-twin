"""Integration tests for the batch-job pipeline (proposal 040).

Test-first: these tests run against services.batch_job.run.main().
They set INPUT_URI / OUTPUT_URI to temporary directories and assert
all three output artifacts are produced and satisfy the documented contracts.

Two scenarios are covered:
  1. Input WITH ground-truth labels (is_anomaly_gt / kind columns present)
     → metrics.json includes ROC-AUC when both classes are present.
  2. Input WITHOUT ground-truth labels (canonical 6 columns only)
     → metrics.json includes a "no_labels" note, no "roc_auc" key.
"""

from __future__ import annotations

import json
import os
import pathlib

import pandas as pd
import pytest

from contracts.schema import ENRICHED_EXTRA_COLUMNS, TRAJECTORY_COLUMNS
from src.data.synthetic import generate
from src.twin.query import ORPHAN_CHECK_QUERY, sparql
from src.twin.store import load_turtle

# Exact column order mandated by contracts/schema.py
_EXPECTED_COLUMNS: list[str] = list(TRAJECTORY_COLUMNS) + list(ENRICHED_EXTRA_COLUMNS)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_labelled_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a small labelled dataset (includes is_anomaly_gt / kind)."""
    df = generate(n_agents=3, n_steps=50, seed=42)
    out = tmp_path / "input_labelled.parquet"
    # Write the full frame (labels included) — read_canonical will select the 6
    # canonical cols; the job reads labels separately via pd.read_parquet.
    df.to_parquet(out, index=False, engine="pyarrow")
    return out


def _write_canonical_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a dataset WITHOUT label columns (6 canonical cols only)."""
    df = generate(n_agents=3, n_steps=50, seed=99)
    out = tmp_path / "input_canonical.parquet"
    df[list(TRAJECTORY_COLUMNS)].to_parquet(out, index=False, engine="pyarrow")
    return out


# ---------------------------------------------------------------------------
# Helper: run the job
# ---------------------------------------------------------------------------

def _run_job(input_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Import and run the batch job with env set to tmp paths."""
    os.environ["INPUT_URI"] = str(input_path)
    os.environ["OUTPUT_URI"] = str(output_dir)
    # Unset MODEL_PATH so load_residual returns the no-op cleanly
    os.environ.pop("MODEL_PATH", None)

    import importlib
    import services.batch_job.run as job_mod
    importlib.reload(job_mod)
    job_mod.main()


# ---------------------------------------------------------------------------
# Scenario 1: labelled input
# ---------------------------------------------------------------------------

def test_labelled_input_produces_all_artifacts(tmp_path):
    """All three artifacts exist; twin.ttl passes orphan-check; metrics valid."""
    input_path = _write_labelled_fixture(tmp_path)
    output_dir = tmp_path / "out_labelled"

    _run_job(input_path, output_dir)

    enriched_path = output_dir / "enriched.parquet"
    twin_ttl = output_dir / "twin.ttl"
    metrics_json = output_dir / "metrics.json"

    assert enriched_path.exists(), "enriched.parquet not found"
    assert twin_ttl.exists(), "twin.ttl not found"
    assert metrics_json.exists(), "metrics.json not found"

    # --- twin.ttl: orphan-check must return zero rows ---
    graph = load_turtle(twin_ttl)
    orphans = sparql(graph, ORPHAN_CHECK_QUERY)
    assert orphans == [], f"Orphan observations found in twin.ttl: {orphans}"

    # --- twin.ttl: non-vacuous (has saref:Observation nodes) ---
    obs_query = (
        "PREFIX saref: <https://saref.etsi.org/core/> "
        "SELECT ?o WHERE { ?o a saref:Observation } LIMIT 1"
    )
    obs_rows = sparql(graph, obs_query)
    assert len(obs_rows) >= 1, "twin.ttl has no saref:Observation nodes (vacuous graph)"

    # --- enriched.parquet: exact contract column order ---
    df_enriched = pd.read_parquet(enriched_path)
    assert list(df_enriched.columns) == _EXPECTED_COLUMNS, (
        f"Column order mismatch.\n"
        f"  Expected: {_EXPECTED_COLUMNS}\n"
        f"  Got:      {list(df_enriched.columns)}"
    )
    assert len(df_enriched) > 0

    # --- pred_lat / pred_lon: populated and in WGS84 bounds ---
    assert df_enriched["pred_lat"].between(-90.0, 90.0).all(), (
        "pred_lat values outside WGS84 bounds"
    )
    assert df_enriched["pred_lon"].between(-180.0, 180.0).all(), (
        "pred_lon values outside WGS84 bounds"
    )

    # --- kind: non-empty for labelled input (passthrough from synthetic generator) ---
    assert (df_enriched["kind"] != "").any(), (
        "kind column is all empty-string in labelled input"
    )
    valid_kinds = {"normal", "sudden_stop", "gps_jump", "geofence_exit"}
    unexpected = set(df_enriched["kind"].unique()) - valid_kinds
    assert not unexpected, f"Unexpected kind values: {unexpected}"

    # --- metrics.json: documented keys ---
    with open(metrics_json) as f:
        metrics = json.load(f)

    required_keys = {"agent_count", "observation_count", "anomaly_count", "runtime_seconds"}
    assert required_keys.issubset(metrics.keys()), (
        f"Missing keys in metrics.json. Have: {set(metrics.keys())}"
    )
    assert metrics["agent_count"] == 3
    assert metrics["observation_count"] > 0
    assert metrics["anomaly_count"] >= 0
    assert metrics["runtime_seconds"] >= 0.0

    # AUC: if present must be valid
    if "roc_auc" in metrics:
        assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_labelled_input_has_correct_per_point_scores(tmp_path):
    """Each observed point carries its own anomaly_score in enriched.parquet."""
    input_path = _write_labelled_fixture(tmp_path)
    output_dir = tmp_path / "out_scores"

    _run_job(input_path, output_dir)

    df_enriched = pd.read_parquet(output_dir / "enriched.parquet")

    # Per-point scores must be valid floats in [0, 1]
    assert df_enriched["anomaly_score"].between(0.0, 1.0).all(), (
        "anomaly_score values out of [0, 1] range"
    )
    # is_anomaly must be boolean-valued
    assert df_enriched["is_anomaly"].isin([True, False]).all()


# ---------------------------------------------------------------------------
# Scenario 1b: FUSE-safe write — stage on local disk, stream-copy to OUTPUT_URI
# ---------------------------------------------------------------------------

def test_artifacts_staged_locally_then_stream_copied(tmp_path, monkeypatch):
    """Artifacts are written to a LOCAL staging dir, then shutil.copyfile'd into
    OUTPUT_URI — never written in-place and never placed via rename/move.

    OUTPUT_URI may be a FUSE object-storage mount; some writers do
    temp-write + rename, which FUSE handles poorly. Staging locally then a pure
    sequential copy (copyfile, not move/rename) dodges that failure mode.
    """
    input_path = _write_canonical_fixture(tmp_path)
    output_dir = tmp_path / "out_staged"
    os.environ["INPUT_URI"] = str(input_path)
    os.environ["OUTPUT_URI"] = str(output_dir)
    os.environ.pop("MODEL_PATH", None)

    import importlib
    import shutil as _shutil
    import services.batch_job.run as job_mod
    importlib.reload(job_mod)

    copied: list[tuple[str, str]] = []
    real_copyfile = _shutil.copyfile

    def spy_copyfile(src, dst, *a, **k):
        copied.append((str(src), str(dst)))
        return real_copyfile(src, dst, *a, **k)

    moved: list = []
    renamed_into_out: list = []
    real_rename = os.rename

    def spy_rename(src, dst, *a, **k):
        if str(output_dir) in str(dst):
            renamed_into_out.append((str(src), str(dst)))
        return real_rename(src, dst, *a, **k)

    monkeypatch.setattr(job_mod.shutil, "copyfile", spy_copyfile)
    monkeypatch.setattr(job_mod.shutil, "move", lambda *a, **k: moved.append(a))
    monkeypatch.setattr(job_mod.os, "rename", spy_rename)

    job_mod.main()

    artifacts = {"enriched.parquet", "twin.ttl", "metrics.json"}
    copied_names = {pathlib.Path(d).name for _, d in copied}
    assert artifacts.issubset(copied_names), (
        f"not all artifacts were copyfile'd into OUTPUT_URI; copied={copied}"
    )
    for src, dst in copied:
        if pathlib.Path(dst).name in artifacts:
            assert pathlib.Path(dst).parent == output_dir, f"copied to wrong dest: {dst}"
            assert pathlib.Path(src).parent != output_dir, (
                f"artifact written in-place, not staged locally: {src}"
            )
    assert not moved, "shutil.move used for placement — rename-based, FUSE-unsafe"
    assert not renamed_into_out, f"os.rename into OUTPUT_URI — FUSE-unsafe: {renamed_into_out}"

    for name in artifacts:
        assert (output_dir / name).exists(), f"{name} missing from OUTPUT_URI"


# ---------------------------------------------------------------------------
# Scenario 2: canonical-only input (no labels)
# ---------------------------------------------------------------------------

def test_canonical_input_no_labels(tmp_path):
    """Input without labels: metrics.json omits roc_auc, has no_labels note;
    kind column is all empty-string; exact column order still holds."""
    input_path = _write_canonical_fixture(tmp_path)
    output_dir = tmp_path / "out_canonical"

    _run_job(input_path, output_dir)

    enriched_path = output_dir / "enriched.parquet"
    twin_ttl = output_dir / "twin.ttl"
    metrics_json = output_dir / "metrics.json"

    assert enriched_path.exists()
    assert twin_ttl.exists()
    assert metrics_json.exists()

    # --- Exact column order even without labels ---
    df_enriched = pd.read_parquet(enriched_path)
    assert list(df_enriched.columns) == _EXPECTED_COLUMNS, (
        f"Column order mismatch (canonical input).\n"
        f"  Expected: {_EXPECTED_COLUMNS}\n"
        f"  Got:      {list(df_enriched.columns)}"
    )

    # --- kind column must be all empty-string when no labels in input ---
    assert (df_enriched["kind"] == "").all(), (
        "kind column should be all '' when input has no 'kind' column"
    )

    # --- metrics.json: no roc_auc, has label_note ---
    with open(metrics_json) as f:
        metrics = json.load(f)

    assert "roc_auc" not in metrics, (
        "roc_auc should be absent when no ground-truth labels are present"
    )
    assert "label_note" in metrics, (
        "Expected 'label_note' key in metrics.json when labels are absent"
    )
    assert "no labels" in metrics["label_note"].lower(), (
        f"label_note does not mention 'no labels': {metrics['label_note']!r}"
    )

    # Orphan check still passes
    graph = load_turtle(twin_ttl)
    orphans = sparql(graph, ORPHAN_CHECK_QUERY)
    assert orphans == []
