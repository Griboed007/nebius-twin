"""Residual model training entry point (optional; baseline is always the fallback).

Usage:
    python -m src.model.train --data-path <parquet> --model-out <path>

The residual net is trained to predict corrections to the baseline.
Training is skipped if torch is not installed; the baseline remains the output.

Designed for a short Nebius Job on L40S:
  - Tiny net (~300 params) → trains in seconds.
  - Saves a state_dict compatible with residual.load_residual().
  - Prints runtime and estimated cost note.

This module is intentionally thin — it imports torch lazily so that
`import src.model.train` does NOT fail in environments without torch.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

logger = logging.getLogger(__name__)


def train(
    data_path: str,
    model_out: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    """Train the tiny residual net on a parquet trajectory dataset.

    Args:
        data_path: Path to parquet file with columns from TRAJECTORY_COLUMNS.
        model_out: Where to save the trained state_dict (.pt).
        epochs:    Number of training epochs.
        lr:        Learning rate.
        batch_size: Mini-batch size.
        seed:      Random seed for reproducibility.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.warning("torch not installed; training skipped. Baseline will be used.")
        return

    torch.manual_seed(seed)

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed; cannot load data.")
        return

    from contracts.schema import HORIZON, HISTORY_LEN, TRAJECTORY_COLUMNS
    from src.model.residual import _TinyResidualNet
    from src.model.baseline import predict_baseline
    from contracts.schema import Trajectory, TrajectoryPoint

    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path)

    net = _TinyResidualNet()
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for agent_id, group in df.groupby("agent_id"):
            group = group.sort_values("t").reset_index(drop=True)
            if len(group) < HISTORY_LEN + HORIZON:
                continue

            # Slide a window over the track
            for start in range(0, len(group) - HISTORY_LEN - HORIZON, HISTORY_LEN // 2):
                obs = group.iloc[start : start + HISTORY_LEN]
                fut = group.iloc[start + HISTORY_LEN : start + HISTORY_LEN + HORIZON]

                history_pts = [
                    TrajectoryPoint(
                        t=float(row.t), lat=float(row.lat), lon=float(row.lon),
                        vlat=float(row.vlat), vlon=float(row.vlon),
                    )
                    for _, row in obs.iterrows()
                ]
                traj = Trajectory(agent_id=str(agent_id), points=history_pts)
                baseline = predict_baseline(traj)

                # Target: residual = future - baseline
                target_dlat = [
                    float(fut.iloc[i].lat) - baseline[i].lat for i in range(HORIZON)
                ]
                target_dlon = [
                    float(fut.iloc[i].lon) - baseline[i].lon for i in range(HORIZON)
                ]
                target = torch.tensor(
                    [val for pair in zip(target_dlat, target_dlon) for val in pair],
                    dtype=torch.float32,
                ).unsqueeze(0)

                vlat = history_pts[-1].vlat
                vlon = history_pts[-1].vlon
                x = torch.tensor([[vlat, vlon]], dtype=torch.float32)

                pred = net(x)
                loss = criterion(pred, target)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d  avg_loss=%.6f", epoch + 1, epochs, avg_loss)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
    torch.save(net.state_dict(), model_out)
    logger.info("Saved residual model to %s", model_out)
    print(f"Training done in {elapsed:.1f}s  →  {model_out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train tiny residual model.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        data_path=args.data_path,
        model_out=args.model_out,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
