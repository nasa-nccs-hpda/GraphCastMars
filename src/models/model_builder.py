# src/models/model_builder.py

import haiku as hk
import xarray as xr
from graphcast import graphcast, normalization, casting, autoregressive, xarray_tree, xarray_jax


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