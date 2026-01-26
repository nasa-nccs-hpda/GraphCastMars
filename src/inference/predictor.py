# src/inference/predictor.py

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging

import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
from graphcast import graphcast
from graphcast import checkpoint
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk

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
    num_steps: int = 10
    lead_time_hours: int = 6
    autoregressive: bool = True
    
    # Model config
    resolution: float = 1.0
    mesh_size: int = 5
    latent_size: int = 512
    gnn_msg_steps: int = 16
    hidden_layers: int = 1
    radius_query_fraction_edge_length: float = 0.6
    
    # Output options
    save_format: str = "netcdf"
    save_intermediate: bool = True
    compress: bool = True
    
    # Visualization
    generate_plots: bool = False
    plot_variables: List[str] = None
    plot_format: str = "png"
    
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
        
        # Flatten nested dicts if needed
        if 'model_config' in config_dict:
            model_config = config_dict.pop('model_config')
            config_dict.update(model_config)
        
        return cls(**config_dict)


class GraphCastPredictor:
    """GraphCast inference predictor"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Load normalization stats
        self.stats = self._load_stats()
        
        # Build model config
        self.model_config, self.task_config = self._build_configs()
        
        # Load model
        self.params, self.state, self.model_fn = self._load_model()
        
        logger.info("GraphCast predictor initialized")
    
    def _load_stats(self) -> Dict[str, xr.Dataset]:
        """Load normalization statistics"""
        stats = {}
        
        stats_files = {
            'mean': 'stats_mean_by_level.nc',
            'stddev': 'stats_stddev_by_level.nc',
            'diffs_stddev': 'stats_diffs_stddev_by_level.nc'
        }
        
        for key, filename in stats_files.items():
            path = self.config.stats_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Stats file not found: {path}")
            stats[key] = xr.open_dataset(path)
            logger.info(f"Loaded {key} statistics from {path}")
        
        return stats
    
    def _build_configs(self) -> Tuple[graphcast.ModelConfig, graphcast.TaskConfig]:
        """Build model and task configurations"""
        # Task config (variables to predict)
        task_config = graphcast.TaskConfig(
            input_variables=[
                '2m_temperature',
                'mean_sea_level_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'temperature',
                'geopotential',
                'u_component_of_wind',
                'v_component_of_wind',
                'vertical_velocity',
                'specific_humidity'
            ],
            target_variables=[
                '2m_temperature',
                'mean_sea_level_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'temperature',
                'geopotential',
                'u_component_of_wind',
                'v_component_of_wind',
                'vertical_velocity',
                'specific_humidity'
            ],
            forcing_variables=['toa_incident_solar_radiation'],
            pressure_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            input_duration='12h'
        )
        
        # Model config
        model_config = graphcast.ModelConfig(
            resolution=self.config.resolution,
            mesh_size=self.config.mesh_size,
            latent_size=self.config.latent_size,
            gnn_msg_steps=self.config.gnn_msg_steps,
            hidden_layers=self.config.hidden_layers,
            radius_query_fraction_edge_length=self.config.radius_query_fraction_edge_length
        )
        
        return model_config, task_config
    
    def _load_model(self) -> Tuple[hk.Params, hk.State, callable]:
        """Load model parameters and create inference function"""
        # Build model
        model = graphcast.GraphCast(self.model_config, self.task_config)
        
        # Load checkpoint
        if not self.config.model_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.config.model_checkpoint}")
        
        with open(self.config.model_checkpoint, 'rb') as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        
        params = ckpt.params
        state = ckpt.state
        
        logger.info(f"Loaded checkpoint from {self.config.model_checkpoint}")
        
        # Create predictor function
        @hk.transform_with_state
        def model_fn(inputs, targets_template, forcings):
            return model(inputs, targets_template, forcings)
        
        # JIT compile
        model_fn_jitted = jax.jit(model_fn.apply)
        
        return params, state, model_fn_jitted
    
    def _load_initial_conditions(self) -> xr.Dataset:
        """Load initial conditions from file or directory"""
        path = self.config.input_data_path
        
        if path.is_file():
            ds = xr.open_dataset(path)
        elif path.is_dir():
            # Load most recent file
            files = sorted(path.glob("*.nc"))
            if not files:
                raise ValueError(f"No NetCDF files found in {path}")
            ds = xr.open_dataset(files[-1])
        else:
            raise ValueError(f"Invalid input path: {path}")
        
        logger.info(f"Loaded initial conditions from {path}")
        return ds
    
    def _normalize_inputs(self, inputs: xr.Dataset) -> xr.Dataset:
        """Normalize inputs using loaded statistics"""
        normalized = normalization.normalize(
            inputs,
            scales=self.stats['mean'],
            locations=self.stats['stddev']
        )
        return normalized
    
    def _denormalize_outputs(self, outputs: xr.Dataset) -> xr.Dataset:
        """Denormalize outputs to physical units"""
        denormalized = normalization.denormalize(
            outputs,
            scales=self.stats['mean'],
            locations=self.stats['stddev']
        )
        return denormalized
    
    def predict_single_step(
        self,
        inputs: xr.Dataset,
        forcings: xr.Dataset
    ) -> Tuple[xr.Dataset, hk.State]:
        """
        Predict single timestep.
        
        Args:
            inputs: Input state (2 timesteps)
            forcings: Forcing variables
            
        Returns:
            (predictions, new_state)
        """
        # Normalize inputs
        inputs_norm = self._normalize_inputs(inputs)
        
        # Create target template (structure for output)
        targets_template = inputs.isel(time=-1)
        
        # Convert to JAX
        inputs_jax = xarray_jax.wrap(inputs_norm)
        forcings_jax = xarray_jax.wrap(forcings)
        targets_template_jax = xarray_jax.wrap(targets_template)
        
        # Run prediction
        predictions_jax, new_state = self.model_fn(
            self.params,
            self.state,
            jax.random.PRNGKey(0),
            inputs_jax,
            targets_template_jax,
            forcings_jax
        )
        
        # Convert back to xarray and denormalize
        predictions = xarray_jax.unwrap(predictions_jax)
        predictions = self._denormalize_outputs(predictions)
        
        return predictions, new_state
    
    def predict(self, initial_conditions: Optional[xr.Dataset] = None) -> List[xr.Dataset]:
        """
        Run multi-step prediction.
        
        Args:
            initial_conditions: Initial state (if None, loads from config)
            
        Returns:
            List of predictions for each timestep
        """
        if initial_conditions is None:
            initial_conditions = self._load_initial_conditions()
        
        predictions = []
        current_inputs = initial_conditions
        current_state = self.state
        
        logger.info(f"Starting {self.config.num_steps}-step prediction")
        
        for step in range(self.config.num_steps):
            logger.info(f"Predicting step {step + 1}/{self.config.num_steps}")
            
            # Extract forcings for this step
            forcings = current_inputs[['toa_incident_solar_radiation']]
            
            # Predict
            prediction, current_state = self.predict_single_step(
                current_inputs,
                forcings
            )
            
            predictions.append(prediction)
            
            # Save intermediate results if requested
            if self.config.save_intermediate:
                output_file = self.config.output_path / f"prediction_step_{step:03d}.nc"
                prediction.to_netcdf(output_file)
                logger.info(f"Saved intermediate prediction to {output_file}")
            
            # Prepare inputs for next step (autoregressive)
            if self.config.autoregressive and step < self.config.num_steps - 1:
                # Use last input and current prediction as new inputs
                new_inputs = xr.concat(
                    [current_inputs.isel(time=-1), prediction],
                    dim='time'
                )
                current_inputs = new_inputs
        
        logger.info("Prediction complete")
        return predictions
    
    def save_predictions(self, predictions: List[xr.Dataset], output_file: Optional[Path] = None):
        """Save all predictions to a single file"""
        if output_file is None:
            output_file = self.config.output_path / "predictions.nc"
        
        # Concatenate all predictions
        combined = xr.concat(predictions, dim='time')
        
        # Save with compression if requested
        encoding = {}
        if self.config.compress:
            for var in combined.data_vars:
                encoding[var] = {'zlib': True, 'complevel': 5}
        
        combined.to_netcdf(output_file, encoding=encoding)
        logger.info(f"Saved predictions to {output_file}")
        
        return output_file