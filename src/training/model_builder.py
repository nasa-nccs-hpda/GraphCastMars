# src/training/model_builder.py

import haiku as hk
import xarray as xr
from graphcast import graphcast, normalization, casting, autoregressive
from typing import Tuple


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
    
    def build_predictor(self):
        """Build normalized GraphCast predictor"""
        
        @hk.transform_with_state
        def forward_fn(inputs, targets_template, forcings):
            predictor = graphcast.GraphCast(self.model_config, self.task_config)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.diffs_stddev_by_level,
                mean_by_level=self.mean_by_level,
                stddev_by_level=self.stddev_by_level
            )
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)
        
        return forward_fn
    
    def build_loss_fn(self):
        """Build loss function"""
        
        @hk.transform_with_state
        def loss_fn(inputs, targets, forcings):
            predictor = graphcast.GraphCast(self.model_config, self.task_config)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.diffs_stddev_by_level,
                mean_by_level=self.mean_by_level,
                stddev_by_level=self.stddev_by_level
            )
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
            return loss.mean(), diagnostics
        
        return loss_fn