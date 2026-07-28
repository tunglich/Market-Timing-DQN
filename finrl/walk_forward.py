"""Expanding-window walk-forward validation utility for FinRL SB3 agents.

Splits a training dataframe into ``n_folds`` chronological segments (default 5)
and runs walk-forward validation with expanding train window:

    Fold 1: train = S1,             val = S2
    Fold 2: train = S1..S2,         val = S3
    Fold 3: train = S1..S3,         val = S4
    Fold 4: train = S1..S4,         val = S5

That is, ``n_folds - 1`` folds when ``n_folds`` segments are used.

The final fold's trained models are also saved to ``final_model_dir`` so that
downstream backtest scripts (which load models by name) can pick them up
without any change.

Usage:
    from walk_forward import run_walk_forward

    metrics = run_walk_forward(
        train_df=train_df,
        env_ctor=lambda df: MyEnv(df=df, **env_kwargs),
        agent_configs={"a2c": None, "ppo": PPO_PARAMS, "ddpg": None,
                       "td3": TD3_PARAMS, "sac": SAC_PARAMS},
        timesteps_per_fold=15_000,
        out_dir=Path("results/my_variant_walkforward"),
        final_model_dir=Path("my_variant_trained_models"),
    )
"""
from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent


def _chronological_segments(df: pd.DataFrame, n_folds: int) -> list[list[str]]:
    """Return ``n_folds`` chronological, roughly equal-sized lists of date
    strings, preserving natural trading-day ordering."""
    unique_dates = sorted(df["date"].unique().tolist())
    n_dates = len(unique_dates)
    if n_dates < n_folds:
        raise ValueError(
            f"Only {n_dates} unique dates available; need at least {n_folds}."
        )
    # np.array_split handles uneven splits gracefully.
    return [list(seg) for seg in np.array_split(unique_dates, n_folds)]


def _filter_by_dates(df: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Return rows of ``df`` whose ``date`` is in ``dates``, reset the integer
    index to a fresh 0-based ``date.factorize``-style index expected by FinRL
    portfolio envs."""
    sub = df[df["date"].isin(dates)].copy()
    sub = sub.sort_values(["date", "tic"]).reset_index(drop=True)
    sub.index = sub.date.factorize()[0]
    return sub


def _rollout_val(
    model,
    val_df: pd.DataFrame,
    env_ctor: Callable[[pd.DataFrame], object],
) -> dict[str, float]:
    """Run the trained agent on the validation env and return summary metrics.

    Returned dict includes:
      - final_value: final account value (in env's currency units)
      - total_return: (final / initial) - 1
      - sharpe: annualized daily Sharpe (mean / std * sqrt(252))
      - max_drawdown: maximum peak-to-trough drawdown of account value
    """
    env_val = env_ctor(val_df)
    df_dret, _ = DRLAgent.DRL_prediction(model=model, environment=env_val)
    df_dret = df_dret.copy()
    r = df_dret["daily_return"].astype(float).to_numpy()
    if r.size == 0:
        return {"final_value": float("nan"), "total_return": float("nan"),
                "sharpe": float("nan"), "max_drawdown": float("nan")}
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    daily_std = float(r.std(ddof=1)) if r.size > 1 else 0.0
    sharpe = float(r.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else float("nan")
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_drawdown = float(dd.min())
    return {
        "final_value": float(equity[-1]),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_walk_forward(
    train_df: pd.DataFrame,
    env_ctor: Callable[[pd.DataFrame], object],
    agent_configs: dict[str, dict | None],
    timesteps_per_fold: int,
    out_dir: Path,
    final_model_dir: Path,
    n_folds: int = 5,
    tb_log_root: Path | None = None,
) -> pd.DataFrame:
    """Run expanding-window walk-forward validation.

    Args:
        train_df: Dataframe covering the full in-sample period. Must have a
            ``date`` column with string dates in ``YYYY-MM-DD`` format.
        env_ctor: Callable that builds a FinRL portfolio env from a dataframe
            (e.g. ``lambda df: CapAnchoredPortfolioEnv(df=df, **env_kwargs)``).
        agent_configs: Mapping ``{agent_name: model_kwargs or None}``. Agent
            names must be recognized by ``DRLAgent.get_model`` (a2c, ppo, ddpg,
            td3, sac).
        timesteps_per_fold: Training timesteps per fold per agent.
        out_dir: Directory to write ``walkforward_metrics.csv``.
        final_model_dir: Directory to save the last fold's models under
            ``agent_<name>.zip`` (compatible with the existing backtest
            scripts).
        n_folds: Number of chronological segments. Yields ``n_folds - 1``
            walk-forward folds.
        tb_log_root: Optional root directory for TensorBoard logs; a subdir
            per fold/agent will be created.

    Returns:
        A DataFrame of per-fold, per-agent validation metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    segments = _chronological_segments(train_df, n_folds)
    print(f"[walk-forward] {n_folds} segments, {n_folds - 1} folds")
    for i, seg in enumerate(segments, 1):
        print(f"  S{i}: {seg[0]} .. {seg[-1]}  ({len(seg)} dates)")

    rows: list[dict] = []
    for fold in range(1, n_folds):
        train_dates = [d for seg in segments[:fold] for d in seg]
        val_dates = segments[fold]
        train_sub = _filter_by_dates(train_df, train_dates)
        val_sub = _filter_by_dates(train_df, val_dates)
        print(
            f"\n=== Fold {fold}/{n_folds - 1} ===\n"
            f"  train: {train_dates[0]} .. {train_dates[-1]}  "
            f"({len(train_dates)} dates, {len(train_sub)} rows)\n"
            f"  val:   {val_dates[0]} .. {val_dates[-1]}    "
            f"({len(val_dates)} dates, {len(val_sub)} rows)"
        )

        env_train = env_ctor(train_sub)
        sb_env_train, _ = env_train.get_sb_env()

        for name, model_kwargs in agent_configs.items():
            tag = f"fold{fold}_{name}"
            print(f"\n  --- Training {name.upper()} ({timesteps_per_fold:,} steps) [{tag}] ---",
                  flush=True)
            t0 = time.time()
            try:
                agent = DRLAgent(env=sb_env_train)
                model = (agent.get_model(name, model_kwargs=model_kwargs)
                         if model_kwargs else agent.get_model(name))
                if tb_log_root is not None:
                    log_dir = tb_log_root / tag
                    log_dir.mkdir(parents=True, exist_ok=True)
                    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
                    model.set_logger(new_logger)
                trained = agent.train_model(model=model, tb_log_name=tag,
                                            total_timesteps=timesteps_per_fold)
                dt_train = time.time() - t0

                metrics = _rollout_val(trained, val_sub, env_ctor)
                dt_val = time.time() - t0 - dt_train
                print(
                    f"    val: return={metrics['total_return'] * 100:+.2f}%  "
                    f"sharpe={metrics['sharpe']:.2f}  "
                    f"mdd={metrics['max_drawdown'] * 100:+.2f}%  "
                    f"(train {dt_train / 60:.1f} min, val {dt_val:.1f}s)",
                    flush=True,
                )
                rows.append({
                    "fold": fold, "agent": name.upper(),
                    "train_start": train_dates[0], "train_end": train_dates[-1],
                    "val_start": val_dates[0], "val_end": val_dates[-1],
                    "train_seconds": dt_train, "timesteps": timesteps_per_fold,
                    **metrics,
                })

                # Save the final fold's models under the standard trained_models
                # location so downstream backtest scripts can load them as-is.
                if fold == n_folds - 1:
                    save_path = final_model_dir / f"agent_{name}"
                    trained.save(str(save_path))
                    print(f"    saved final-fold model: {save_path}.zip", flush=True)
            except Exception:
                dt_train = time.time() - t0
                print(f"    {name.upper()} FAILED after {dt_train / 60:.1f} min", flush=True)
                traceback.print_exc()
                rows.append({
                    "fold": fold, "agent": name.upper(),
                    "train_start": train_dates[0], "train_end": train_dates[-1],
                    "val_start": val_dates[0], "val_end": val_dates[-1],
                    "train_seconds": dt_train, "timesteps": timesteps_per_fold,
                    "final_value": float("nan"), "total_return": float("nan"),
                    "sharpe": float("nan"), "max_drawdown": float("nan"),
                })

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "walkforward_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\n[walk-forward] metrics saved to {metrics_path}")
    return metrics_df
