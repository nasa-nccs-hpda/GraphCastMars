# src/inference/predictor.py

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging
import functools
import sys
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
    autoregressive: bool = True
    target_lead_times: str = "6h"
    
    
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
            #config_dict['model_config'] = ModelConfig(**model_config_dict)
        
        return cls(**config_dict)



class GraphCastPredictor:
    """GraphCast inference predictor - based on original DeepMind implementation"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Load checkpoint (includes model_config and task_config)
        self.ckpt = self._load_checkpoint()
        self.params = self.ckpt.params
        self.state = {}
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
            
            target_lead_times = slice(self.config.target_lead_times,
                                      f"{6*self.config.num_steps}h")
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                ds,
                target_lead_times=target_lead_times,
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

    def predict_single_file(self, input_file: Path) -> xr.Dataset:
        """Run prediction on single input data"""

        if not input_file.exists():
            raise FileNotFoundError(f"Input data not found: {input_file}")
        
        inputs, targets, forcings = self._load_initial_conditions(input_file)
        
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
        
        logger.info("Prediction complete for {input_file.name}")
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
        input_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        file_pattern: str = "*.nc"
    ) -> List[Path]:
        """
        Run prediction and save results. Handles both single files and directories.
        
        Args:
            input_path: Path to input file or directory (if None, uses config)
            output_dir: Output directory (if None, uses config)
            file_pattern: Glob pattern for files in directory (default: "*.nc")
            
        Returns:
            List of paths to saved prediction files
        """
        # Use config paths if not provided
        if input_path is None:
            input_path = self.config.input_data_path
        else:
            input_path = Path(input_path)
        
        if output_dir is None:
            output_dir = self.config.output_path
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if input is file or directory
        if input_path.is_file():
            # Single file case
            logger.info(f"Processing single file: {input_path}")
            
            predictions = self.predict_single_file(input_path)
            
            # Generate output filename based on input
            output_file = output_dir / f"{input_path.stem}_prediction.nc"
            saved_path = self.save_predictions(predictions, output_file)
            
            return [saved_path]
        
        elif input_path.is_dir():
            # Directory case - process all matching files
            logger.info(f"Processing directory: {input_path}")
            logger.info(f"  File pattern: {file_pattern}")
            
            # Find all matching files
            input_files = sorted(input_path.glob(file_pattern))
            
            if not input_files:
                raise ValueError(f"No files matching '{file_pattern}' found in {input_path}")
            
            logger.info(f"Found {len(input_files)} files to process")
            
            saved_paths = []
            
            for i, input_file in enumerate(input_files, 1):
                logger.info(f"\n[{i}/{len(input_files)}] Processing: {input_file.name}")
                
                try:
                    # Run prediction
                    predictions = self.predict_single_file(input_file)
                    
                    # Generate output filename
                    output_file = output_dir / f"{input_file.stem}_prediction.nc"
                    
                    # Save
                    saved_path = self.save_predictions(predictions, output_file)
                    saved_paths.append(saved_path)
                    
                except Exception as e:
                    logger.error(f"Failed to process {input_file.name}: {e}")
                    continue
            
            logger.info(f"\nCompleted: {len(saved_paths)}/{len(input_files)} files processed successfully")
            
            return saved_paths
        
        else:
            raise ValueError(f"Input path must be a file or directory: {input_path}")


def main():
    """
    Main entry point for running predictions from command line.
    
    Usage:
        python -m src.inference.predictor
        python -m src.inference.predictor --config configs/custom_inference.yaml
    """
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run GraphCast predictions on Mars data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference.yaml',
        help='Path to inference configuration file (default: configs/inference.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = InferenceConfig.from_yaml(Path(args.config))
        
        # Display configuration
        print("\n" + "="*60)
        print("GraphCast Mars Inference")
        print("="*60)
        print(f"Model checkpoint:  {config.model_checkpoint}")
        print(f"Stats directory:   {config.stats_dir}")
        print(f"Input data:        {config.input_data_path}")
        print(f"Output directory:  {config.output_path}")
        print(f"Prediction steps:  {config.num_steps}")
        print(f"Lead time:         {config.target_lead_times} per step")
        print("="*60)
        print()
        
        # Initialize predictor
        logger.info("Initializing predictor...")
        predictor = GraphCastPredictor(config)
        
        # Run predictions
        logger.info("Running predictions...")
        output_files = predictor.predict_and_save()
        
        # Report results
        print()
        print("="*60)
        print("✓ Prediction Complete!")
        print("="*60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())