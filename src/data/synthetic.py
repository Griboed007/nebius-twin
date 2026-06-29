"""Deterministic synthetic GPS trajectory generator.

Usage (CLI):
    python -m src.data.synthetic --out artifacts/synthetic.parquet --seed 7

The generator produces N agents of road-like GPS tracks (piecewise-linear
headings, gaussian position noise) keyed by seed so two runs with identical
seed + parameters are byte-identical.

Three anomaly types are injected with a parallel ground-truth label column:
  - ``sudden_stop``:   velocity drops to ~0 for several consecutive steps
  - ``gps_jump``:      position jumps by a large delta within WGS84 bounds
  - ``geofence_exit``: agent moves outside the nominal bounding box

Ground-truth label columns (NOT in TRAJECTORY_COLUMNS; travel alongside):
  ``is_anomaly_gt`` (bool) — True for every point belonging to an anomaly event
  ``kind``           (str) — "normal" | "sudden_stop" | "gps_jump" | "geofence_exit"

Join key to enriched predictions (020+): (agent_id, t).  These deliberately
differ from the *predicted* ``is_anomaly`` in ENRICHED_EXTRA_COLUMNS to avoid
a name collision when joining ground-truth against predictions.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

import numpy as np
import pandas as pd

from contracts.schema import TRAJECTORY_COLUMNS, SAMPLE_PERIOD_S

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Allowed anomaly-kind values (exported for consumer validation)
ANOMALY_KINDS: tuple[str, ...] = ("sudden_stop", "gps_jump", "geofence_exit")

# Base region: Moscow metropolitan area (well inside WGS84 bounds)
_BASE_LAT: float = 55.75
_BASE_LON: float = 37.61
_GEOFENCE_RADIUS_DEG: float = 0.15  # ~16 km in lat/lon degrees

# Noise parameters
_POS_NOISE_STD: float = 0.0001   # degrees, ~11 m
_VEL_STD: float = 0.00005        # degrees/s baseline velocity magnitude

# Anomaly injection parameters
_STOP_DURATION_STEPS: int = 5
_JUMP_DELTA_DEG: float = 0.08    # large but stays inside WGS84
_GEOFENCE_EXIT_DELTA_DEG: float = _GEOFENCE_RADIUS_DEG + 0.03


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate(
    n_agents: int = 5,
    n_steps: int = 200,
    seed: int = 42,
    anomaly_rate: float = 0.05,
    base_lat: float = _BASE_LAT,
    base_lon: float = _BASE_LON,
    sample_period: float = SAMPLE_PERIOD_S,
) -> pd.DataFrame:
    """Generate synthetic GPS trajectories with labelled anomalies.

    Parameters
    ----------
    n_agents:
        Number of independent agents.
    n_steps:
        Number of time steps per agent.
    seed:
        RNG seed; identical seed + params → byte-identical output.
    anomaly_rate:
        Target fraction of anomalous points across the whole dataset.
        Achieved within ±tolerance (guaranteed only if n_agents*n_steps is
        large enough to fit at least one of each anomaly type).
    base_lat, base_lon:
        Centre of the simulated region (WGS84 decimal degrees).
    sample_period:
        Seconds between observations.

    Returns
    -------
    DataFrame with columns:
        agent_id, t, lat, lon, vlat, vlon  (TRAJECTORY_COLUMNS)
        is_anomaly_gt (bool), kind (str)
    """
    rng = np.random.default_rng(seed)

    total_points = n_agents * n_steps
    # Determine how many anomalous points we need in total.
    # Guarantee at least _STOP_DURATION_STEPS per anomaly type (×3) so each
    # type appears at least once even on a tiny dataset.
    min_anomaly_points = len(ANOMALY_KINDS) * _STOP_DURATION_STEPS
    target_anomaly_points = max(min_anomaly_points, int(round(anomaly_rate * total_points)))

    rows: list[dict] = []

    # --- Build each agent's trajectory independently ---
    for agent_idx in range(n_agents):
        agent_id = f"agent_{agent_idx:03d}"
        lat = base_lat + rng.uniform(-0.05, 0.05)
        lon = base_lon + rng.uniform(-0.05, 0.05)

        # Piecewise-linear heading: change heading every segment_len steps
        segment_len = max(10, n_steps // 8)
        heading = rng.uniform(0, 2 * np.pi)

        speed = rng.uniform(0.00005, 0.00015)  # degrees/s

        vlat = speed * np.cos(heading)
        vlon = speed * np.sin(heading)

        track_lats = np.zeros(n_steps)
        track_lons = np.zeros(n_steps)
        track_vlats = np.zeros(n_steps)
        track_vlons = np.zeros(n_steps)
        track_kinds: list[str] = ["normal"] * n_steps

        current_lat = lat
        current_lon = lon
        current_vlat = vlat
        current_vlon = vlon

        for step in range(n_steps):
            # Possibly rotate heading at segment boundaries
            if step > 0 and step % segment_len == 0:
                delta = rng.uniform(-np.pi / 4, np.pi / 4)
                heading = heading + delta
                speed = rng.uniform(0.00005, 0.00015)
                current_vlat = speed * np.cos(heading)
                current_vlon = speed * np.sin(heading)

            # Gaussian position noise
            noise_lat = rng.normal(0, _POS_NOISE_STD)
            noise_lon = rng.normal(0, _POS_NOISE_STD)

            track_lats[step] = current_lat + noise_lat
            track_lons[step] = current_lon + noise_lon
            track_vlats[step] = current_vlat
            track_vlons[step] = current_vlon

            # Advance position
            current_lat += current_vlat * sample_period
            current_lon += current_vlon * sample_period

        agent_rows = {
            "lat": track_lats,
            "lon": track_lons,
            "vlat": track_vlats,
            "vlon": track_vlons,
            "kind": track_kinds,
        }
        rows.append((agent_id, n_steps, agent_rows))

    # --- Distribute anomaly budget across agents and types ---
    # We cycle through anomaly types and assign to random agents/positions.
    # Use the pre-seeded rng for reproducibility.
    anomaly_type_cycle = list(ANOMALY_KINDS)
    # Count how many anomaly events of each type to inject
    points_per_type = target_anomaly_points // len(ANOMALY_KINDS)
    events_per_type = max(1, points_per_type // _STOP_DURATION_STEPS)

    for type_idx, kind in enumerate(anomaly_type_cycle):
        for _ in range(events_per_type):
            agent_idx = int(rng.integers(0, n_agents))
            agent_id, agent_n_steps, agent_rows = rows[agent_idx]
            # Pick a start position that fits the event
            max_start = agent_n_steps - _STOP_DURATION_STEPS - 1
            if max_start < 1:
                continue
            start = int(rng.integers(1, max_start))
            end = start + _STOP_DURATION_STEPS

            if kind == "sudden_stop":
                # Zero velocity for the event window
                agent_rows["vlat"][start:end] = 0.0
                agent_rows["vlon"][start:end] = 0.0
                for s in range(start, end):
                    agent_rows["kind"][s] = "sudden_stop"

            elif kind == "gps_jump":
                # Large position offset applied to the midpoint step,
                # clamped to WGS84 bounds
                mid = start + _STOP_DURATION_STEPS // 2
                for s in range(start, end):
                    new_lat = float(np.clip(
                        agent_rows["lat"][s] + _JUMP_DELTA_DEG,
                        -90.0, 90.0,
                    ))
                    new_lon = float(np.clip(
                        agent_rows["lon"][s] + _JUMP_DELTA_DEG,
                        -180.0, 180.0,
                    ))
                    agent_rows["lat"][s] = new_lat
                    agent_rows["lon"][s] = new_lon
                    agent_rows["kind"][s] = "gps_jump"

            elif kind == "geofence_exit":
                # Push position outside the geofence radius
                for s in range(start, end):
                    new_lat = float(np.clip(
                        agent_rows["lat"][s] + _GEOFENCE_EXIT_DELTA_DEG,
                        -90.0, 90.0,
                    ))
                    new_lon = float(np.clip(
                        agent_rows["lon"][s] + _GEOFENCE_EXIT_DELTA_DEG,
                        -180.0, 180.0,
                    ))
                    agent_rows["lat"][s] = new_lat
                    agent_rows["lon"][s] = new_lon
                    agent_rows["kind"][s] = "geofence_exit"

    # --- Assemble DataFrame ---
    all_rows = []
    t0 = 0.0
    for agent_idx, (agent_id, agent_n_steps, agent_rows) in enumerate(rows):
        t_start = t0 + agent_idx * agent_n_steps * sample_period
        ts = t_start + np.arange(agent_n_steps) * sample_period
        all_rows.append(pd.DataFrame({
            "agent_id": agent_id,
            "t": ts,
            "lat": agent_rows["lat"],
            "lon": agent_rows["lon"],
            "vlat": agent_rows["vlat"],
            "vlon": agent_rows["vlon"],
            "is_anomaly_gt": [k != "normal" for k in agent_rows["kind"]],
            "kind": agent_rows["kind"],
        }))

    df = pd.concat(all_rows, ignore_index=True)

    # Enforce exact dtypes for byte-stability
    df["agent_id"] = df["agent_id"].astype(str)
    df["t"] = df["t"].astype(np.float64)
    df["lat"] = df["lat"].astype(np.float64)
    df["lon"] = df["lon"].astype(np.float64)
    df["vlat"] = df["vlat"].astype(np.float64)
    df["vlon"] = df["vlon"].astype(np.float64)
    df["is_anomaly_gt"] = df["is_anomaly_gt"].astype(bool)
    df["kind"] = df["kind"].astype(str)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GPS trajectories and write to parquet."
    )
    parser.add_argument(
        "--out",
        default="artifacts/synthetic.parquet",
        help="Output parquet path (default: artifacts/synthetic.parquet)",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (default: 7)")
    parser.add_argument(
        "--n-agents", type=int, default=10, help="Number of agents (default: 10)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=300, help="Steps per agent (default: 300)"
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Target anomaly fraction (default: 0.05)",
    )
    args = parser.parse_args(argv)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        seed=args.seed,
        anomaly_rate=args.anomaly_rate,
    )

    # Write the full labelled frame (includes is_anomaly_gt and kind)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # Write a separate canonical parquet (TRAJECTORY_COLUMNS only) as sidecar
    from src.data.io import write_canonical
    canonical_path = out_path.with_suffix("").with_name(out_path.stem + ".canonical.parquet")
    write_canonical(df, canonical_path)

    total = len(df)
    n_anomaly = int(df["is_anomaly_gt"].sum())
    print(f"Written {total} rows ({n_anomaly} anomalous, {n_anomaly/total:.1%}) to {out_path}")
    print(f"Canonical (6-col) sidecar: {canonical_path}")
    for kind in ANOMALY_KINDS:
        count = int((df["kind"] == kind).sum())
        print(f"  {kind}: {count} rows")


if __name__ == "__main__":
    _cli_main()
