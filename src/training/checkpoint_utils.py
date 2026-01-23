# src/training/checkpoint_utils.py

import xarray as xr
from pathlib import Path
from typing import Dict, Any
from graphcast import checkpoint, graphcast


def load_checkpoint(checkpoint_path: str) -> graphcast.CheckPoint:
    """Load GraphCast checkpoint"""
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return ckpt


def load_normalization_stats(stats_dir: str):
    """Load normalization statistics"""
    stats_dir = Path(stats_dir)
    
    with open(stats_dir / "stats_mean_by_level.nc", "rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    
    with open(stats_dir / "stats_stddev_by_level.nc", "rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()
    
    with open(stats_dir / "stats_diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    
    return mean_by_level, stddev_by_level, diffs_stddev_by_level


def save_checkpoint(
    path: str,
    params: Dict,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    description: str = "Fine-tuned GraphCast for Mars"
):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    ckpt = graphcast.CheckPoint(
        params=params,
        model_config=model_config,
        task_config=task_config,
        description=description,
        license="Apache 2.0"
    )
    
    with open(path, 'wb') as f:
        checkpoint.dump(f, ckpt)