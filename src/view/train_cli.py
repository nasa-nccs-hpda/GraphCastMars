# src/view/train_cli.py

import click
from pathlib import Path
import logging

from ..training.trainer import GraphCastTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group(name='train')
def train_group():
    """Commands for training GraphCast models"""
    pass


@train_group.command(name='run')
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              required=True, help='Training configuration file')
def train_run(config: Path):
    """
    Train GraphCast model on MCD data.
    
    Example:
        graphcast-mars train run --config configs/training_config.yaml
    """
    try:
        # Load config
        training_config = TrainingConfig.from_yaml(config)
        
        # Create trainer
        trainer = GraphCastTrainer(training_config)
        
        # Train
        trainer.train()
        
        click.echo("✅ Training complete!")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("Training failed")
        raise click.Abort()


@train_group.command(name='generate-config')
@click.option('--output', type=click.Path(path_type=Path), required=True)
def train_generate_config(output: Path):
    """Generate training configuration template"""
    import yaml
    
    config_template = {
        'checkpoint_path': '/path/to/params_GraphCast_small.npz',
        'stats_dir': '/path/to/stats',
        'data_dir': '/path/to/mcd_data',
        'output_dir': './output/training',
        'num_epochs': 100,
        'batch_size': 1,
        'learning_rate': 0.0001,
        'weight_decay': 0.01,
        'gradient_clip': 5.0,
        'train_split': 0.8,
        'patience': 10,
        'save_every': 10
    }
    
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False)
    
    click.echo(f"✅ Config template saved to: {output}")