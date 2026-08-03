"""Centralised seeding for the FinRL 8-indicator variants (Figure 8 basis).

The 5 SB3 agents (A2C, PPO, DDPG, TD3, SAC) each pull random samples from
Python, NumPy, and PyTorch RNGs during training. To make Figure 8's overlays
reproducible we set every RNG **before** the environment is constructed.

Usage in a train/backtest script (2 lines near the top of the file, after
imports but before you construct the environment):

    from finrl_seeds import set_finrl_seeds, read_seed_from_env
    SEED = set_finrl_seeds(read_seed_from_env(default=42))

The default seed is 42, overridable via ``FINRL_SEED`` env var:

    export FINRL_SEED=17          # bash / zsh
    $env:FINRL_SEED = "17"        # PowerShell

Note that SB3's ``PPO/A2C/...`` constructors accept a ``seed`` kwarg but the
default FinRL wrapper (``DRLAgent.get_model``) does not forward one. Seeding the
process-wide RNGs before ``get_model`` runs is sufficient in practice because
SB3 draws its initial weights from the global torch RNG.
"""
from __future__ import annotations

import os
import random


def read_seed_from_env(default: int = 42) -> int:
    """Return ``int(os.environ["FINRL_SEED"])`` or ``default`` if unset/invalid."""
    raw = os.environ.get("FINRL_SEED", "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[finrl.seeds] Ignoring FINRL_SEED='{raw}' (not int); using {default}.")
        return default


def set_finrl_seeds(seed: int) -> int:
    """Seed Python/NumPy/PyTorch (and SB3 if available). Returns the seed."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(seed)
    except ImportError:
        pass
    print(f"[finrl.seeds] Seeded RNGs with seed={seed}")
    return seed
