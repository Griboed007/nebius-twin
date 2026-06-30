"""Batch job entry point — proposal 040.

Reads a trajectory parquet from INPUT_URI, scores each agent point-by-point,
builds an RDF twin graph, and writes three artifacts to OUTPUT_URI:
  enriched.parquet  — per-point anomaly scores for all agents
  twin.ttl          — semantic digital twin in Turtle format
  metrics.json      — run statistics (agent count, observations, AUC if labels)

All paths are env-driven. No hardcoded paths. Runs on CPU by default.

Usage (module form, as wired in the Makefile):
    python3 -m services.batch_job.run

Env vars:
    INPUT_URI   — path to labelled or canonical trajectory parquet
                  (default: ./artifacts/synthetic.parquet)
    OUTPUT_URI  — directory prefix for output artifacts
                  (default: ./artifacts/out)
    MODEL_PATH  — optional path to a residual .pt file; omit for baseline-only

enriched.parquet column contract (contracts/schema.py):
    list(TRAJECTORY_COLUMNS) + list(ENRICHED_EXTRA_COLUMNS)
    = [agent_id, t, lat, lon, vlat, vlon, pred_lat, pred_lon, anomaly_score, is_anomaly, kind]

  pred_lat / pred_lon — first forward step of the per-point predict() call
    (prediction.points[0].lat/lon).  predict_baseline always returns HORIZON
    points even for single-point windows (stationary projection), so
    prediction.points[0] is always defined; no NaN fallback is needed.
    For 1-point windows the stationary projection equals the observed lat/lon.

  kind — passthrough of the input 'kind' column if present (010 synthetic label:
    "normal"|"sudden_stop"|"gps_jump"|"geofence_exit"), else "" (empty string).
    This is the ground-truth anomaly-type label from the input, NOT the RDF
    kind ("observed"|"predicted") used in twin.ttl.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import tempfile
import time
from typing import Optional

import pandas as pd

from contracts.schema import (
    ANOMALY_THRESHOLD,
    ENRICHED_EXTRA_COLUMNS,
    TRAJECTORY_COLUMNS,
    EnrichedPoint,
    EnrichedTrajectory,
    Prediction,
    Trajectory,
    TrajectoryPoint,
)
from src.data.io import read_canonical
from src.model.core import predict
from src.model.residual import load_residual
from src.twin.graph import build_graph
from src.twin.store import save_turtle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Canonical enriched column order (imported from contract — never hardcoded).
_ENRICHED_COLUMNS: list[str] = list(TRAJECTORY_COLUMNS) + list(ENRICHED_EXTRA_COLUMNS)


# ---------------------------------------------------------------------------
# Per-point scoring
# ---------------------------------------------------------------------------

def _score_per_point(
    agent_id: str,
    points: list[TrajectoryPoint],
    residual_model,
) -> tuple[list[EnrichedPoint], list[tuple[float, float]]]:
    """Score each observed point using a growing window of history.

    For point i, predict() is called ONCE on points[:i+1].  The same call
    produces both the anomaly score (via retrodiction) and the one-step-ahead
    forward prediction used for pred_lat/pred_lon.  No extra predict() calls.

    Returns
    -------
    (enriched_points, pred_positions)
      enriched_points  — list[EnrichedPoint] for building EnrichedTrajectory
      pred_positions   — list[(pred_lat, pred_lon)] aligned by index to enriched_points

    Edge windows:
      - 1-point window: predict_baseline returns a stationary projection
        (pred lat/lon equals the observed lat/lon); anomaly_score=0.0.
      - Windows 2–4 points: baseline extrapolates, retrodiction returns 0.0
        (too short to split RETRO_STEPS=3 tail + 2-point prefix).
      predict() handles all short-window cases internally; no special casing here.
    """
    enriched: list[EnrichedPoint] = []
    pred_positions: list[tuple[float, float]] = []

    for i in range(len(points)):
        window = points[: i + 1]
        traj_window = Trajectory(agent_id=agent_id, points=window)
        pred = predict(traj_window, residual_model=residual_model)

        score = pred.anomaly_score
        # pred.points is guaranteed non-empty (predict_baseline always returns HORIZON)
        pred_lat = pred.points[0].lat
        pred_lon = pred.points[0].lon

        enriched.append(
            EnrichedPoint(
                t=points[i].t,
                lat=points[i].lat,
                lon=points[i].lon,
                vlat=points[i].vlat,
                vlon=points[i].vlon,
                anomaly_score=score,
                is_anomaly=score >= ANOMALY_THRESHOLD,
            )
        )
        pred_positions.append((pred_lat, pred_lon))

    return enriched, pred_positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: reads env, runs pipeline, writes artifacts."""
    t_start = time.monotonic()

    # --- Env ---
    input_uri = os.environ.get("INPUT_URI", "./artifacts/synthetic.parquet")
    output_uri = os.environ.get("OUTPUT_URI", "./artifacts/out")
    model_path: Optional[str] = os.environ.get("MODEL_PATH") or None

    input_path = pathlib.Path(input_uri)
    output_path = pathlib.Path(output_uri)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("INPUT_URI=%s", input_uri)
    logger.info("OUTPUT_URI=%s", output_uri)
    logger.info("MODEL_PATH=%s", model_path)

    # --- Load residual ONCE at job start (020 purity contract) ---
    residual = load_residual(model_path)
    logger.info("Residual model: %s", type(residual).__name__)

    # --- Read canonical trajectory columns ---
    df_canonical = read_canonical(input_path)

    # --- Attempt to read optional ground-truth label columns ---
    df_full = pd.read_parquet(input_path, engine="pyarrow")
    has_labels = "is_anomaly_gt" in df_full.columns
    has_kind = "kind" in df_full.columns

    if has_labels:
        logger.info("Ground-truth labels detected (is_anomaly_gt / kind).")
    else:
        logger.info("No ground-truth labels in input; AUC will not be computed.")

    # --- Per-agent enrichment ---
    enriched_trajectories: list[EnrichedTrajectory] = []
    enriched_rows: list[dict] = []
    # For AUC computation: accumulate (score, gt_label) pairs aligned by (agent_id, t)
    score_pairs: list[tuple[float, bool]] = []

    agent_ids = df_canonical["agent_id"].unique()
    logger.info("Processing %d agents", len(agent_ids))

    for agent_id in sorted(agent_ids):
        agent_canonical_df = (
            df_canonical[df_canonical["agent_id"] == agent_id].sort_values("t")
        )
        agent_full_df = (
            df_full[df_full["agent_id"] == agent_id].sort_values("t")
        )

        # Build TrajectoryPoints from canonical frame
        points = [
            TrajectoryPoint(
                t=float(row.t),
                lat=float(row.lat),
                lon=float(row.lon),
                vlat=float(row.vlat),
                vlon=float(row.vlon),
            )
            for row in agent_canonical_df.itertuples(index=False)
        ]

        if not points:
            logger.warning("Agent %s has no points; skipping.", agent_id)
            continue

        # Per-point anomaly scoring + one-step-ahead pred position (same predict() call)
        observed, pred_positions = _score_per_point(agent_id, points, residual)

        # Forward prediction over the full history (for EnrichedTrajectory.prediction)
        full_traj = Trajectory(agent_id=agent_id, points=points)
        forward_pred: Prediction = predict(full_traj, residual_model=residual)

        enriched_traj = EnrichedTrajectory(
            agent_id=agent_id,
            observed=observed,
            prediction=forward_pred,
        )
        enriched_trajectories.append(enriched_traj)

        # Collect rows for enriched.parquet
        # kind — passthrough from input if column present, else ""
        kind_values = (
            agent_full_df["kind"].values if has_kind else None
        )
        gt_series = (
            agent_full_df["is_anomaly_gt"].values if has_labels else None
        )

        for i, pt in enumerate(observed):
            pred_lat, pred_lon = pred_positions[i]
            kind_val = (
                str(kind_values[i]) if kind_values is not None and i < len(kind_values)
                else ""
            )
            enriched_rows.append(
                {
                    "agent_id": agent_id,
                    "t": pt.t,
                    "lat": pt.lat,
                    "lon": pt.lon,
                    "vlat": pt.vlat,
                    "vlon": pt.vlon,
                    "pred_lat": pred_lat,
                    "pred_lon": pred_lon,
                    "anomaly_score": pt.anomaly_score,
                    "is_anomaly": pt.is_anomaly,
                    "kind": kind_val,
                }
            )

            # Accumulate score/label pairs for AUC (aligned by sorted position)
            if gt_series is not None and i < len(gt_series):
                score_pairs.append((pt.anomaly_score, bool(gt_series[i])))

    # --- Stage all artifacts on LOCAL disk, then stream-copy to OUTPUT_URI ---
    # OUTPUT_URI may be a FUSE object-storage mount (Nebius bucket). pyarrow and
    # rdflib can write temp-then-rename, which object-storage FUSE handles poorly.
    # So write everything to a local staging dir first (fast POSIX, real rename),
    # then place each artifact onto OUTPUT_URI with shutil.copyfile — a pure
    # sequential write with no rename. Local OUTPUT_URI just gets a cheap extra copy.
    stage_dir = pathlib.Path(tempfile.mkdtemp(prefix="twinjob-"))
    try:
        # enriched.parquet (exact contract column order)
        df_enriched = pd.DataFrame(enriched_rows, columns=_ENRICHED_COLUMNS)
        df_enriched.to_parquet(stage_dir / "enriched.parquet", index=False, engine="pyarrow")
        logger.info(
            "Staged enriched.parquet (%d rows, columns=%s)",
            len(df_enriched),
            list(df_enriched.columns),
        )

        # twin.ttl
        graph = build_graph(enriched_trajectories)
        save_turtle(graph, stage_dir / "twin.ttl")
        logger.info("Staged twin.ttl (%d triples)", len(graph))

        # metrics.json
        anomaly_count = int(df_enriched["is_anomaly"].sum()) if len(df_enriched) > 0 else 0
        runtime_seconds = time.monotonic() - t_start
        metrics: dict = {
            "agent_count": len(enriched_trajectories),
            "observation_count": len(df_enriched),
            "anomaly_count": anomaly_count,
            "runtime_seconds": round(runtime_seconds, 3),
        }
        if has_labels:
            _add_auc(metrics, score_pairs)
        else:
            metrics["label_note"] = "no labels — ROC-AUC not computed"
        with open(stage_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Stream-copy each staged artifact onto OUTPUT_URI (no rename → FUSE-safe).
        for name in ("enriched.parquet", "twin.ttl", "metrics.json"):
            shutil.copyfile(stage_dir / name, output_path / name)
            logger.info("Copied %s -> %s", name, output_path / name)
        logger.info("Wrote metrics.json: %s", metrics)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _add_auc(metrics: dict, score_pairs: list[tuple[float, bool]]) -> None:
    """Compute ROC-AUC and add to metrics, guarding against single-class data."""
    if not score_pairs:
        metrics["label_note"] = "no labels — ROC-AUC not computed"
        return

    scores = [s for s, _ in score_pairs]
    labels = [int(g) for _, g in score_pairs]

    if len(set(labels)) < 2:
        # sklearn roc_auc_score raises ValueError on single-class y_true
        metrics["label_note"] = (
            "only one class present in ground-truth labels — ROC-AUC undefined"
        )
        return

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, scores))
        metrics["roc_auc"] = round(auc, 4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROC-AUC computation failed: %s", exc)
        metrics["label_note"] = f"ROC-AUC computation failed: {exc}"


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
