# src/training/data_loader.py

import glob
import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterator
from graphcast import graphcast


class MarsDataLoader:
    """Load and batch MCD data for training"""
    
    def __init__(self, data_dir: str, batch_size: int = 1):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
    
    def get_file_list(self, pattern: str = "*.nc") -> List[Path]:
        """Get list of data files"""
        files = sorted(glob.glob(str(self.data_dir / pattern)))
        return [Path(f) for f in files]
    
    def split_train_val(self, files: List[Path], train_split: float = 0.8) -> Tuple[List[Path], List[Path]]:
        """Split files into train/validation"""
        split_idx = int(len(files) * train_split)
        return files[:split_idx], files[split_idx:]
    
    def load_sample(self, file_path: Path, task_config: graphcast.TaskConfig):
        """Load single training sample"""
        ds = xr.open_dataset(file_path)
        
        # Extract inputs (first 2 timesteps)
        inputs = ds.isel(time=slice(0, 2))
        
        # Extract targets (timestep 2)
        targets = ds.isel(time=slice(2, 3))
        
        # Forcings (all timesteps)
        forcings = ds
        
        return inputs, targets, forcings
    
    def data_iterator(self, files: List[Path], task_config: graphcast.TaskConfig, shuffle: bool = True) -> Iterator:
        """Create data iterator"""
        if shuffle:
            files = np.random.permutation(files).tolist()
        
        for file_path in files:
            inputs, targets, forcings = self.load_sample(file_path, task_config)
            yield inputs, targets, forcings