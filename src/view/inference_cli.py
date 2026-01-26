# src/view/inference_cli.py

import click
from pathlib import Path
from typing import Optional
import logging

from ..inference.predictor import InferenceConfig, GraphCastPredictor
from ..inference.postprocessing import PredictionVisualizer

logger = logging.getLogger(__name__)


@click.group(name='predict')
def predict_group():
    """Commands for running GraphCast predictions"""
    pass


@predict_group.command(name='run')
@click.option('--config', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--input', type=click.Path(exists=True, path_type=Path), help='Override input data path')
@click.option('--output', type=click.Path(path_type=Path), help='Override output directory')
@click.option('--num-steps', type=int, help='Override number of prediction steps')
@click.option('--visualize/--no-visualize', default=None, help='Generate visualizations')
def predict_run(config: Path, input: Optional[Path], output: Optional[Path], 
                num_steps: Optional[int], visualize: Optional[bool]):
    """
    Run GraphCast prediction.
    
    Example:
        graphcast-mars predict run --config inference_config.yaml
        graphcast-mars predict run --config inference_config.yaml --num-steps 20 --visualize
    """
    try:
        # Load config
        inf_config = InferenceConfig.from_yaml(config)
        
        # Override with CLI args
        if input:
            inf_config.input_data_path = input
        if output:
            inf_config.output_path = output
        if num_steps:
            inf_config.num_steps = num_steps
        if visualize is not None:
            inf_config.generate_plots = visualize
        
        click.echo("📋 Inference Configuration")
        click.echo("=" * 50)
        click.echo(f"Model checkpoint:  {inf_config.model_checkpoint}")
        click.echo(f"Input data:        {inf_config.input_data_path}")
        click.echo(f"Output directory:  {inf_config.output_path}")
        click.echo(f"Prediction steps:  {inf_config.num_steps}")
        click.echo(f"Lead time:         {inf_config.lead_time_hours}h per step")
        click.echo()
        
        # Initialize predictor
        click.echo("🔧 Initializing predictor...")
        predictor = GraphCastPredictor(inf_config)
        
        # Run prediction
        click.echo("🚀 Running prediction...")
        predictions = predictor.predict()
        
        # Save results
        click.echo("💾 Saving predictions...")
        output_file = predictor.save_predictions(predictions)
        
        click.echo()
        click.echo(f"✅ Prediction complete!")
        click.echo(f"📊 Generated {len(predictions)} timesteps")
        click.echo(f"📁 Saved to: {output_file}")
        
        # Visualize
        if inf_config.generate_plots:
            click.echo("\n📊 Generating visualizations...")
            visualizer = PredictionVisualizer(inf_config.output_path / "plots")
            
            # Load predictions
            ds = xr.open_dataset(output_file)
            
            for var in inf_config.plot_variables or ['2m_temperature']:
                if var in ds:
                    for t in range(min(5, len(ds.time))):
                        visualizer.plot_variable(ds, var, timestep=t)
            
            click.echo("✅ Visualizations saved")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("Error during prediction")
        raise click.Abort()


@predict_group.command(name='generate-config')
@click.option('--output', type=click.Path(path_type=Path), required=True)
def predict_generate_config(output: Path):
    """Generate inference configuration template"""
    import yaml
    
    config = {
        'model_checkpoint': '/path/to/checkpoint.npz',
        'stats_dir': '/path/to/stats/',
        'input_data_path': '/path/to/initial_conditions.nc',
        'output_path': './predictions',
        'num_steps': 10,
        'lead_time_hours': 6,
        'autoregressive': True,
        'model_config': {
            'resolution': 1.0,
            'mesh_size': 5,
            'latent_size': 512,
            'gnn_msg_steps': 16
        },
        'generate_plots': True,
        'plot_variables': ['2m_temperature', 'geopotential']
    }
    
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    click.echo(f"✅ Config template saved to: {output}")