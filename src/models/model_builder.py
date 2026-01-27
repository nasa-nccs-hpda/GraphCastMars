# src/models/model_builder.py
import jax
import jax.numpy as jnp
import haiku as hk
import xarray as xr
from typing import Tuple, Dict, Optional, Callable
from graphcast import graphcast, normalization, casting, autoregressive, xarray_tree, xarray_jax
import functools
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
class ModelBuilder:
    """Build and configure GraphCast models"""
    
    def __init__(
        self,
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig,
        mean_by_level: xr.Dataset,
        stddev_by_level: xr.Dataset,
        diffs_stddev_by_level: xr.Dataset
    ):
        self.model_config = model_config
        self.task_config = task_config
        self.mean_by_level = mean_by_level
        self.stddev_by_level = stddev_by_level
        self.diffs_stddev_by_level = diffs_stddev_by_level
    
    def construct_wrapped_graphcast(self, model_config, task_config):
        """Constructs and wraps the GraphCast Predictor"""
        # Base predictor
        predictor = graphcast.GraphCast(model_config, task_config)
        
        # Add BFloat16 casting
        predictor = casting.Bfloat16Cast(predictor)
        
        # Add normalization (casting happens AFTER normalization)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=self.diffs_stddev_by_level,
            mean_by_level=self.mean_by_level,
            stddev_by_level=self.stddev_by_level
        )
        
        # Make autoregressive
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        
        return predictor
    
    def build_predictor(self):
        """Build normalized GraphCast predictor for inference"""
        
        @hk.transform_with_state
        def forward_fn(model_config, task_config, inputs, targets_template, forcings):
            predictor = self.construct_wrapped_graphcast(model_config, task_config)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)
        
        return forward_fn
    
    def build_loss_fn(self):
        """Build loss function (returns mean loss and diagnostics)"""
        
        @hk.transform_with_state
        def loss_fn(model_config, task_config, inputs, targets, forcings):
            predictor = self.construct_wrapped_graphcast(model_config, task_config)
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
            
            # Take mean and unwrap to JAX arrays
            return xarray_tree.map_structure(
                lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
                (loss, diagnostics)
            )
        
        return loss_fn
    
    def build_jitted_predictor(
        self,
        params: hk.Params,
        state: hk.State
    ) -> Callable:
        """
        Build JIT-compiled predictor with params and state bound.
        
        This creates a predictor ready for inference, matching the original
        DeepMind implementation pattern.
        
        Args:
            params: Model parameters
            state: Model state
            
        Returns:
            JIT-compiled predictor function
        """
        forward_fn = self.build_predictor()
        
        # Helper to pass configs via partial (as in original)
        def with_configs(fn):
            return functools.partial(
                fn,
                model_config=self.model_config,
                task_config=self.task_config
            )
        
        # Helper to pass params and state
        def with_params(fn):
            return functools.partial(fn, params=params, state=state)
        
        # Drop state from output (models aren't stateful)
        def drop_state(fn):
            return lambda **kw: fn(**kw)[0]
        
        # JIT compile
        predictor_jitted = drop_state(
            with_params(
                jax.jit(
                    with_configs(forward_fn.apply)
                )
            )
        )
        
        return predictor_jitted
    
class NormalizationManager:
    """Manage normalization/denormalization (shared between training and inference)"""
    
    def __init__(self, stats_dir: Path):
        self.stats_dir = Path(stats_dir)
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict[str, xr.Dataset]:
        """Load normalization statistics"""
        stats = {}
        
        stats_files = {
            'mean': 'stats_mean_by_level.nc',
            'stddev': 'stats_stddev_by_level.nc',
            'diffs_stddev': 'stats_diffs_stddev_by_level.nc'
        }
        
        for key, filename in stats_files.items():
            path = self.stats_dir / filename
            if not path.exists():
                logger.warning(f"Stats file not found: {path}")
                continue
            stats[key] = xr.open_dataset(path)
            logger.info(f"Loaded {key} statistics from {path}")
        
        return stats
    
    def normalize(self, data: xr.Dataset) -> xr.Dataset:
        """Normalize data using loaded statistics"""
        if 'mean' not in self.stats or 'stddev' not in self.stats:
            logger.warning("Missing normalization stats, returning unnormalized data")
            return data
        
        normalized = normalization.normalize(
            data,
            scales=self.stats['stddev'],
            locations=self.stats['mean']
        )
        return normalized
    
    def denormalize(self, data: xr.Dataset) -> xr.Dataset:
        """Denormalize data to physical units"""
        if 'mean' not in self.stats or 'stddev' not in self.stats:
            logger.warning("Missing normalization stats, returning data as-is")
            return data
        
        denormalized = normalization.denormalize(
            data,
            scales=self.stats['stddev'],
            locations=self.stats['mean']
        )
        return denormalized