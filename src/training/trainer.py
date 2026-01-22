# src/training/trainer.py

import jax
import optax
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

from .model_builder import ModelBuilder
from .data_loader import MarsDataLoader
from .checkpoint_utils import load_checkpoint, load_normalization_stats, save_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    checkpoint_path: str
    stats_dir: str
    data_dir: str
    output_dir: str
    
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 5.0
    
    train_split: float = 0.8
    patience: int = 10
    save_every: int = 10
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class GraphCastTrainer:
    """GraphCast training orchestrator"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint and stats
        self.ckpt = load_checkpoint(config.checkpoint_path)
        self.params = self.ckpt.params
        self.state = {}
        self.model_config = self.ckpt.model_config
        self.task_config = self.ckpt.task_config
        
        self.mean_by_level, self.stddev_by_level, self.diffs_stddev_by_level = \
            load_normalization_stats(config.stats_dir)
        
        # Build model
        self.model_builder = ModelBuilder(
            self.model_config, self.task_config,
            self.mean_by_level, self.stddev_by_level, self.diffs_stddev_by_level
        )
        
        self.forward_fn = self.model_builder.build_predictor()
        self.loss_fn = self.model_builder.build_loss_fn()
        
        # JIT compile
        self.loss_fn_apply = jax.jit(self.loss_fn.apply)
        
        # Setup optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            optax.adamw(
                learning_rate=config.learning_rate,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                weight_decay=config.weight_decay
            )
        )
        self.opt_state = self.optimizer.init(self.params)
        
        # Data loader
        self.data_loader = MarsDataLoader(config.data_dir, config.batch_size)
    
    def train_step(self, inputs, targets, forcings):
        """Single training step"""
        
        def compute_loss(params):
            (loss, diagnostics), new_state = self.loss_fn_apply(
                params, self.state, jax.random.PRNGKey(0),
                inputs, targets, forcings
            )
            return loss, (diagnostics, new_state)
        
        (loss, (diagnostics, new_state)), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(self.params)
        
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        
        return new_params, new_state, new_opt_state, loss, diagnostics
    
    def val_step(self, inputs, targets, forcings):
        """Single validation step"""
        (loss, diagnostics), _ = self.loss_fn_apply(
            self.params, self.state, jax.random.PRNGKey(0),
            inputs, targets, forcings
        )
        return loss
    
    def train(self):
        """Main training loop"""
        files = self.data_loader.get_file_list()
        train_files, val_files = self.data_loader.split_train_val(files, self.config.train_split)
        
        logger.info(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            epoch_losses = []
            for inputs, targets, forcings in self.data_loader.data_iterator(
                train_files, self.task_config, shuffle=True
            ):
                self.params, self.state, self.opt_state, loss, diagnostics = \
                    self.train_step(inputs, targets, forcings)
                
                loss_value = float(loss)
                epoch_losses.append(loss_value)
                
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}, Loss: {loss_value:.6f}")
                
                if (global_step + 1) % self.config.save_every == 0:
                    ckpt_path = f"{self.config.output_dir}/checkpoint_step_{global_step:05d}.npz"
                    save_checkpoint(ckpt_path, self.params, self.model_config, self.task_config)
                
                global_step += 1
            
            avg_train_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.6f}")
            
            # Validation
            val_losses = []
            for inputs, targets, forcings in self.data_loader.data_iterator(
                val_files, self.task_config, shuffle=False
            ):
                val_loss = self.val_step(inputs, targets, forcings)
                val_losses.append(float(val_loss))
            
            avg_val_loss = np.mean(val_losses)
            logger.info(f"Epoch {epoch + 1} - Avg Val Loss: {avg_val_loss:.6f}")
            
            # Best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_path = f"{self.config.output_dir}/best_model.npz"
                save_checkpoint(best_path, self.params, self.model_config, self.task_config)
                logger.info(f"New best model! Val loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info("Training complete!")