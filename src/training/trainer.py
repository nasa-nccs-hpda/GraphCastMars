# src/training/trainer.py

import jax
import optax
import numpy as np
import functools
from pathlib import Path
from dataclasses import dataclass
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
    save_every: int = 10
    target_lead_times: str = "6h"
    
    @classmethod
    def from_yaml(cls, config_path: str):
        import yaml
        with open(config_path, 'r') as f:
            return cls(**yaml.safe_load(f))


def grads_fn(params, state, model_config, task_config, loss_fn, inputs, targets, forcings):
    """Compute loss, diagnostics, and gradients"""
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f
        )
        return loss, (diagnostics, next_state)
    
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True
    )(params, state, inputs, targets, forcings)
    
    return loss, diagnostics, next_state, grads


def with_configs(fn, model_config, task_config, loss_fn):
    """Bind configs to function using functools.partial"""
    return functools.partial(fn, model_config=model_config, task_config=task_config, loss_fn=loss_fn)


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
        
        self.loss_fn = self.model_builder.build_loss_fn()
        
        # JIT compile grads function
        self.grads_fn_jitted = jax.jit(
            with_configs(grads_fn, self.model_config, self.task_config, self.loss_fn)
        )
        
        # Setup optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            optax.adamw(learning_rate=config.learning_rate, b1=0.9, b2=0.999, 
                       eps=1e-8, weight_decay=config.weight_decay)
        )
        self.opt_state = self.optimizer.init(self.params)
        
        # Data loader
        target_lead_times = slice(config.target_lead_times, config.target_lead_times)
        self.data_loader = MarsDataLoader(config.data_dir, config.batch_size, target_lead_times)
    
    def train(self):
        """Main training loop"""
        files = self.data_loader.get_file_list()
        logger.info(f"Found {len(files)} training files")
        
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_losses = []
            
            for inputs, targets, forcings in self.data_loader.data_iterator(
                files, self.task_config, shuffle=True
            ):
                # Compute gradients
                loss, diagnostics, next_state, grads = self.grads_fn_jitted(
                    params=self.params, state=self.state,
                    inputs=inputs, targets=targets, forcings=forcings
                )
                
                # Update parameters
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
                self.params = optax.apply_updates(self.params, updates)
                self.state = next_state
                
                loss_value = float(loss)
                epoch_losses.append(loss_value)
                
                # Logging
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}, Loss: {loss_value:.6f}")
                
                if global_step % 50 == 0:
                    print("diagnostics:", diagnostics)
                    print("loss:", loss)
                
                # Checkpointing
                if (global_step + 1) % self.config.save_every == 0:
                    ckpt_path = f"{self.config.output_dir}/checkpoint_step_{global_step:05d}.npz"
                    save_checkpoint(ckpt_path, self.params, self.model_config, self.task_config)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
                
                # Save best model
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_path = f"{self.config.output_dir}/best_model.npz"
                    save_checkpoint(best_path, self.params, self.model_config, self.task_config)
                    logger.info(f"New best model! Loss: {best_loss:.6f}")
                
                global_step += 1
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.6f}")
        
        logger.info("Training complete!")


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Train GraphCast on Mars data")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    args = parser.parse_args()
    
    config = TrainingConfig.from_yaml(args.config)
    trainer = GraphCastTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()