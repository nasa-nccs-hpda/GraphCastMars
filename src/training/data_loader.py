# src/training/data_loader.py

import xarray
import dataclasses
from pathlib import Path
from typing import List, Tuple, Iterator
import logging

from graphcast import data_utils, graphcast

logger = logging.getLogger(__name__)


class MarsDataLoader:
    """Load and batch MCD data for training using GraphCast's data utilities"""
    
    def __init__(self, data_dir: str, batch_size: int = 1, target_lead_times=slice("6h", "6h")):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.target_lead_times = target_lead_times
    
    def get_file_list(self, pattern: str = "*.nc") -> List[Path]:
        """Get list of data files"""
        import glob
        files = sorted(glob.glob(str(self.data_dir / pattern)))
        return [Path(f) for f in files]
    
    def split_train_val(self, files: List[Path], train_split: float = 0.8) -> Tuple[List[Path], List[Path]]:
        """Split files into train/validation"""
        split_idx = int(len(files) * train_split)
        return files[:split_idx], files[split_idx:]
    
    def extract_example(
        self, 
        file_path: Path, 
        task_config: graphcast.TaskConfig
    ) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
        """
        Extract inputs, targets, and forcings from a single example file.
        
        Uses GraphCast's data_utils.extract_inputs_targets_forcings which handles:
        - Proper timestep extraction
        - Normalization preparation
        - Batch dimension handling
        """
        try:
            with open(file_path, "rb") as f:
                ds = xarray.load_dataset(f).compute()
            
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                ds,
                target_lead_times=self.target_lead_times,
                **dataclasses.asdict(task_config)
            )
            
            logger.debug(f"Loaded: {file_path.name}")
            logger.debug(f"  Inputs dims: {inputs.dims.mapping}")
            logger.debug(f"  Targets dims: {targets.dims.mapping}")
            logger.debug(f"  Forcings dims: {forcings.dims.mapping}")
            
            return (inputs, targets, forcings)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def collate_batch(
        self, 
        batch: List[Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]]
    ) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
        """
        Collate a batch of examples.
        
        For batch_size=1, just returns the single example.
        For batch_size>1, concatenates along batch dimension.
        """
        if len(batch) == 1:
            return batch[0]
        
        # Separate inputs, targets, forcings
        inputs_list = [item[0] for item in batch]
        targets_list = [item[1] for item in batch]
        forcings_list = [item[2] for item in batch]
        
        # Concatenate along batch dimension
        batched_inputs = xarray.concat(inputs_list, dim='batch')
        batched_targets = xarray.concat(targets_list, dim='batch')
        batched_forcings = xarray.concat(forcings_list, dim='batch')
        
        return batched_inputs, batched_targets, batched_forcings
    
    def data_iterator(
        self, 
        files: List[Path], 
        task_config: graphcast.TaskConfig, 
        shuffle: bool = True
    ) -> Iterator[Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]]:
        """
        Create data iterator that yields batches.
        
        Args:
            files: List of data file paths
            task_config: GraphCast task configuration
            shuffle: Whether to shuffle files
            
        Yields:
            (inputs, targets, forcings) tuples for each batch
        """
        import numpy as np
        
        if shuffle:
            files = np.random.permutation(files).tolist()
        
        batch = []
        
        for file_path in files:
            try:
                example = self.extract_example(file_path, task_config)
                batch.append(example)
                
                # Yield batch when full
                if len(batch) == self.batch_size:
                    yield self.collate_batch(batch)
                    logger.debug(f"Yielded batch from: {file_path.name}")
                    batch = []
                    
            except Exception as e:
                logger.warning(f"Skipping file {file_path}: {e}")
                continue
        
        # Yield remaining partial batch if any
        if batch:
            logger.debug(f"Yielding partial batch of size {len(batch)}")
            yield self.collate_batch(batch)


# Standalone functions for backward compatibility
def extract_example(
    file_path: str, 
    task_config: graphcast.TaskConfig, 
    target_lead_times=slice("6h", "6h")
) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """
    Standalone function to extract inputs, targets, and forcings.
    
    Args:
        file_path: Path to data file
        task_config: GraphCast task configuration
        target_lead_times: Target prediction lead times
        
    Returns:
        (inputs, targets, forcings) tuple
    """
    with open(file_path, "rb") as f:
        ds = xarray.load_dataset(f).compute()
    
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds,
        target_lead_times=target_lead_times,
        **dataclasses.asdict(task_config)
    )
    
    print("Inputs Paths:", file_path)
    print("Batched Inputs:", inputs.dims.mapping)
    
    return (inputs, targets, forcings)


def batch_data_loader(
    file_list: List[str], 
    task_config: graphcast.TaskConfig, 
    batch_size: int = 1, 
    target_lead_times=slice("6h", "6h")
) -> Iterator[Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]]:
    """
    Generator to yield batches of inputs, targets, and forcings.
    
    Args:
        file_list: List of file paths
        task_config: GraphCast task configuration
        batch_size: Number of examples per batch
        target_lead_times: Target prediction lead times
        
    Yields:
        (inputs, targets, forcings) tuples for each batch
    """
    batch = []
    
    for file in file_list:
        example = extract_example(file, task_config, target_lead_times)
        batch.append(example)
        
        if len(batch) == batch_size:
            # For batch_size=1, just yield the single example
            if batch_size == 1:
                yield batch[0]
            else:
                # For larger batches, concatenate
                inputs_list = [item[0] for item in batch]
                targets_list = [item[1] for item in batch]
                forcings_list = [item[2] for item in batch]
                
                batched_inputs = xarray.concat(inputs_list, dim='batch')
                batched_targets = xarray.concat(targets_list, dim='batch')
                batched_forcings = xarray.concat(forcings_list, dim='batch')
                
                yield (batched_inputs, batched_targets, batched_forcings)
            
            print(file)
            batch = []
    
    # Yield remaining partial batch if any
    if batch:
        if batch_size == 1:
            yield batch[0]
        else:
            inputs_list = [item[0] for item in batch]
            targets_list = [item[1] for item in batch]
            forcings_list = [item[2] for item in batch]
            
            batched_inputs = xarray.concat(inputs_list, dim='batch')
            batched_targets = xarray.concat(targets_list, dim='batch')
            batched_forcings = xarray.concat(forcings_list, dim='batch')
            
            yield (batched_inputs, batched_targets, batched_forcings)