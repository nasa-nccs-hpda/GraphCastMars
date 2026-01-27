# src/inference/predictor.py

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging
import functools

import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import dataclasses as dc

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree

from ..models.model_builder import ModelBuilder, NormalizationManager
from ..models.checkpoint_utils import load_checkpoint, load_normalization_stats

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for GraphCast inference"""
    # Model paths
    model_checkpoint: Path
    stats_dir: Path
    
    # Input/output
    input_data_path: Path
    output_path: Path
    
    # Prediction settings
    num_steps: int = 4  # Number of forecast steps (eval_steps in original)
    lead_time_hours: int = 6
    autoregressive: bool = True
    
    
    # Output options
    save_format: str = "netcdf"
    save_intermediate: bool = False
    compress: bool = True
    
    # Chunking (for memory efficiency)
    use_chunked_prediction: bool = True
    
    def __post_init__(self):
        self.model_checkpoint = Path(self.model_checkpoint)
        self.stats_dir = Path(self.stats_dir)
        self.input_data_path = Path(self.input_data_path)
        self.output_path = Path(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'InferenceConfig':
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract model config if present
        if 'model_config' in config_dict:
            model_config_dict = config_dict.pop('model_config')
            config_dict['model_config'] = ModelConfig(**model_config_dict)
        
        return cls(**config_dict)



class GraphCastPredictor:
    """GraphCast inference predictor - based on original DeepMind implementation"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Load checkpoint (includes model_config and task_config)
        self.ckpt = self._load_checkpoint()
        self.params = self.ckpt.params
        self.state = self.ckpt.state
        self.model_config = self.ckpt.model_config
        self.task_config = self.ckpt.task_config
        
        logger.info(f"Loaded checkpoint from {config.model_checkpoint}")
        logger.info(f"Model resolution: {self.model_config.resolution}°")
        logger.info(f"Description: {self.ckpt.description}")
        
        # Load normalization statistics
        self.normalizer = NormalizationManager(config.stats_dir)
        
        # Build model builder with loaded stats
        self.model_builder = ModelBuilder(
            model_config=self.model_config,
            task_config=self.task_config,
            mean_by_level=self.normalizer.stats['mean'],
            stddev_by_level=self.normalizer.stats['stddev'],
            diffs_stddev_by_level=self.normalizer.stats['diffs_stddev']
        )
        
        # Build JIT-compiled predictor
        self.run_forward_jitted = self.model_builder.build_jitted_predictor(
            self.params,
            self.state
        )
        
        logger.info("GraphCast predictor initialized")
    
    def _load_checkpoint(self) -> graphcast.CheckPoint:
        """Load checkpoint file"""
        if not self.config.model_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.config.model_checkpoint}")
        
        ckpt = load_checkpoint(self.config.model_checkpoint)
        
        return ckpt
    
    def _load_initial_conditions(self, file_path) -> xr.Dataset:
        """Load initial conditions from file or directory"""
        try:
            with open(file_path, "rb") as f:
                ds = xr.load_dataset(f).compute()
            
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                ds,
                target_lead_times=self.target_lead_times,
                **dc.asdict(self.task_config)
            )
            
            logger.info(f"Loaded: {file_path.name}")
            logger.info(f"  Inputs dims: {inputs.dims.mapping}")
            logger.info(f"  Targets dims: {targets.dims.mapping}")
            logger.info(f"  Forcings dims: {forcings.dims.mapping}")
            
            return (inputs, targets, forcings)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def predict(self) -> xr.Dataset:
        """Run prediction on input data"""
        path = self.config.input_data_path

        if not path.exists():
            raise FileNotFoundError(f"Input data not found: {path}")
        
        inputs, targets, forcings = self._load_initial_conditions(path)
        
        # Create targets template (filled with NaN as in original)
        targets_template = targets * np.nan
        
        logger.info(f"Starting {self.config.num_steps}-step prediction")
        
        # Run prediction
        if self.config.use_chunked_prediction:
            predictions = rollout.chunked_prediction(
                self.run_forward_jitted,
                rng=jax.random.PRNGKey(0),
                inputs=inputs,
                targets_template=targets_template,
                forcings=forcings
            )
        else:
            # Alternative: direct prediction (may use more memory)
            predictions = self.run_forward_jitted(
                rng=jax.random.PRNGKey(0),
                inputs=inputs,
                targets_template=targets_template,
                forcings=forcings
            )
        
        logger.info("Prediction complete")
        logger.info(f"  Output dimensions: {predictions.dims.mapping}")
        
        return predictions
        

    def save_predictions(
        self, 
        predictions: xr.Dataset, 
        output_file: Optional[Path] = None,
        source_label: str = "era5-mcd"
    ) -> Path:
        """
        Save predictions to NetCDF file.
        
        Args:
            predictions: Predictions dataset
            output_file: Output file path (if None, auto-generate)
            source_label: Label for output filename
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = predictions.time.values[0]
            output_file = self.config.output_path / f"fm_graphcast_{source_label}_output.nc"
        
        # Add metadata
        predictions.attrs['model'] = 'GraphCast-Mars'
        predictions.attrs['checkpoint'] = str(self.config.model_checkpoint)
        predictions.attrs['num_steps'] = self.config.num_steps
        predictions.attrs['description'] = self.ckpt.description
        
        # Save with compression
        encoding = {}
        if self.config.compress:
            for var in predictions.data_vars:
                encoding[var] = {'zlib': True, 'complevel': 5}
        
        predictions.to_netcdf(output_file, encoding=encoding)
        logger.info(f"Saved predictions to {output_file}")
        
        return output_file

    def predict_and_save(
        self, 
        input_data: Optional[xr.Dataset] = None,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Convenience method: predict and save in one call.
        
        Args:
            input_data: Input dataset (if None, loads from config)
            output_file: Output file path (if None, auto-generate)
            
        Returns:
            Path to saved predictions
        """
        predictions = self.predict(input_data)
        output_path = self.save_predictions(predictions, output_file)
        return output_path


# Convenience function for simple usage
def run_inference(
    checkpoint_path: Path,
    stats_dir: Path,
    input_data_path: Path,
    output_path: Path,
    num_steps: int = 4
) -> Path:
    """
    Simple inference function that mirrors original script usage.
    
    Example:
        >>> output = run_inference(
        ...     checkpoint_path=Path("checkpoints/params_GraphCast_small.npz"),
        ...     stats_dir=Path("checkpoints/graphcast/"),
        ...     input_data_path=Path("data/source-era5-mcd_date-2022-01-01.nc"),
        ...     output_path=Path("predictions/"),
        ...     num_steps=4
        ... )
    """
    config = InferenceConfig(
        model_checkpoint=checkpoint_path,
        stats_dir=stats_dir,
        input_data_path=input_data_path,
        output_path=output_path,
        num_steps=num_steps
    )
    
    predictor = GraphCastPredictor(config)
    return predictor.predict_and_save()


# Example usage matching original script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Paths (matching original script structure)
    checkpoint_path = Path("/discover/nobackup/jli30/systest/Graphcast_Mars_test/checkpoints/graphcast/params_GraphCast_small.npz")
    stats_dir = Path("/discover/nobackup/jli30/systest/Graphcast_Mars_test/checkpoints/graphcast")
    input_data = Path("/discover/nobackup/jli30/systest/Graphcast_Mars_test/format_out")
    output_dir = Path("/discover/nobackup/jli30/systest/Graphcast_Mars_test/pred_out")
    
    # Run inference
    output_file = run_inference(
        checkpoint_path=checkpoint_path,
        stats_dir=stats_dir,
        input_data_path=input_data,
        output_path=output_dir,
        num_steps=4
    )
    
    print(f"✅ Predictions saved to {output_file}")