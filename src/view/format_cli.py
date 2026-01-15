# src/view/format_cli.py

import click
from pathlib import Path
from typing import Optional
import logging

from ..preprocessing.graphcast_formatter import GraphCastFormatterConfig, GraphCastFormatter

logger = logging.getLogger(__name__)


@click.group(name='format')
def format_group():
    """Commands for formatting GraphCast input data"""
    pass


@format_group.command(name='run')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to formatter configuration file'
)
@click.option(
    '--date',
    help='Process single date (YYYY-MM-DD), otherwise process all dates in config'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be processed without actually processing'
)
def format_run(config: Path, date: Optional[str], dry_run: bool):
    """
    Format MCD data for GraphCast input.
    
    Combines MCD variables with ERA5 template according to configuration.
    
    Example:
        graphcast-mars format run --config format_config.yaml
        graphcast-mars format run --config format_config.yaml --date 2022-03-20
    """
    try:
        # Load config
        click.echo(f"Loading configuration from: {config}")
        formatter_config = GraphCastFormatterConfig.from_yaml(config)
        
        click.echo()
        click.echo("📋 Configuration Summary")
        click.echo("=" * 50)
        click.echo(f"MCD data:          {formatter_config.mcd_data_path}")
        click.echo(f"ERA5 samples:      {formatter_config.era5_sample_path}")
        click.echo(f"ERA5 stats:        {formatter_config.era5_stats_path}")
        click.echo(f"Output:            {formatter_config.output_path}")
        click.echo(f"Experiment:        {formatter_config.experiment_name}")
        click.echo()
        
        if date:
            click.echo(f"📅 Processing single date: {date}")
        else:
            click.echo(f"📅 Processing date range: {formatter_config.start_date} + {formatter_config.num_days} days")
        
        click.echo()
        click.echo("🔧 Variable Strategies:")
        
        # Group strategies by type
        strategies = {
            'mcd': [],
            'era5_mean': [],
            'constant': [],
            'keep_era5': []
        }
        
        for vs in formatter_config.variable_strategies:
            strategies[vs.strategy].append(vs.name)
        
        if strategies['mcd']:
            click.echo(f"   📊 From MCD ({len(strategies['mcd'])}):")
            for var in strategies['mcd']:
                click.echo(f"      • {var}")
        
        if strategies['era5_mean']:
            click.echo(f"   📈 ERA5 mean ({len(strategies['era5_mean'])}):")
            for var in strategies['era5_mean']:
                click.echo(f"      • {var}")
        
        if strategies['constant']:
            click.echo(f"   🔢 Constants ({len(strategies['constant'])}):")
            for var in strategies['constant']:
                vs = next(v for v in formatter_config.variable_strategies if v.name == var)
                click.echo(f"      • {var} = {vs.constant_value}")
        
        if strategies['keep_era5']:
            click.echo(f"   ♻️  Keep ERA5 ({len(strategies['keep_era5'])}):")
            for var in strategies['keep_era5']:
                click.echo(f"      • {var}")
        
        if dry_run:
            click.echo()
            click.echo("🔍 Dry run mode - no files will be generated")
            return
        
        click.echo()
        click.echo("🚀 Starting formatting...")
        
        # Create formatter
        formatter = GraphCastFormatter(formatter_config)
        
        # Process
        if date:
            output_files = formatter.process_single_date(date)
            click.echo()
            click.echo(f"✅ Processed {date}")
            click.echo(f"📊 Generated {len(output_files)} files")
            for f in output_files:
                click.echo(f"   • {f.name}")
        else:
            results = formatter.process_all_dates()
            total_files = sum(len(v) for v in results.values())
            successful_dates = sum(1 for v in results.values() if v)
            
            click.echo()
            click.echo(f"✅ Processing complete!")
            click.echo(f"📊 Processed {successful_dates}/{len(results)} dates")
            click.echo(f"📁 Generated {total_files} files total")
            click.echo(f"📂 Output directory: {formatter_config.output_path.absolute()}")
        
    except FileNotFoundError as e:
        click.echo(f"❌ File not found: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("Error during formatting")
        raise click.Abort()


@format_group.command(name='generate-config')
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    required=True,
    help='Path for config template file'
)
@click.option(
    '--template',
    type=click.Choice(['temperature-only', 'multi-variable', 'custom']),
    default='temperature-only',
    help='Configuration template type [default: temperature-only]'
)
def format_generate_config(output: Path, template: str):
    """
    Generate a template configuration file for GraphCast formatting.
    
    Templates:
      - temperature-only: Only MCD temperature variables
      - multi-variable: Multiple MCD variables (temperature, wind, etc.)
      - custom: Minimal template for customization
    
    Example:
        graphcast-mars format generate-config --output format_config.yaml --template temperature-only
    """
    try:
        # Create base config
        config = GraphCastFormatterConfig(
            mcd_data_path="/path/to/mcd/data",
            era5_sample_path="/path/to/era5/samples",
            era5_stats_path="/path/to/era5/stats.nc",
            output_path="./output",
            experiment_name="mcd_experiment"
        )
        
        # Customize based on template
        if template == 'temperature-only':
            # Keep default (temperature only)
            pass
        
        elif template == 'multi-variable':
            # Add more MCD variables
            from ..preprocessing.graphcast_formatter import VariableStrategy
            
            # Add wind variables from MCD
            for vs in config.variable_strategies:
                if vs.name in ['10m_u_component_of_wind', '10m_v_component_of_wind',
                              'u_component_of_wind', 'v_component_of_wind']:
                    vs.strategy = 'mcd'
                    vs.scale_to_era5 = True
        
        elif template == 'custom':
            # Minimal config
            config.variable_strategies = []
        
        config.to_yaml(output)
        
        click.echo(f"✅ Config template saved to: {output.absolute()}")
        click.echo(f"📝 Template type: {template}")
        click.echo()
        click.echo("Next steps:")
        click.echo(f"  1. Edit {output} to customize settings")
        click.echo(f"  2. Run: graphcast-mars format run --config {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@format_group.command(name='info')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to formatter config file'
)
def format_info(config: Path):
    """
    Display information about a formatting configuration.
    
    Example:
        graphcast-mars format info --config format_config.yaml
    """
    try:
        formatter_config = GraphCastFormatterConfig.from_yaml(config)
        
        click.echo("📋 GraphCast Formatter Configuration")
        click.echo("=" * 60)
        click.echo()
        
        click.echo("📂 Paths:")
        click.echo(f"   MCD data:          {formatter_config.mcd_data_path}")
        click.echo(f"   ERA5 samples:      {formatter_config.era5_sample_path}")
        click.echo(f"   ERA5 stats:        {formatter_config.era5_stats_path}")
        click.echo(f"   Output:            {formatter_config.output_path}")
        click.echo()
        
        click.echo("⚙️  Settings:")
        click.echo(f"   Experiment:        {formatter_config.experiment_name}")
        click.echo(f"   Start date:        {formatter_config.start_date}")
        click.echo(f"   Number of days:    {formatter_config.num_days}")
        click.echo(f"   Time step:         {formatter_config.time_step_hours} hours")
        click.echo(f"   Input steps:       {formatter_config.num_input_steps}")
        click.echo(f"   Output steps:      {formatter_config.num_output_steps}")
        click.echo(f"   Resolution:        {formatter_config.target_resolution}°")
        click.echo()
        
        # Variable strategies summary
        strategies = {
            'mcd': [],
            'era5_mean': [],
            'constant': [],
            'keep_era5': []
        }
        
        for vs in formatter_config.variable_strategies:
            strategies[vs.strategy].append(vs)
        
        click.echo("📊 Variable Strategies:")
        
        if strategies['mcd']:
            click.echo(f"\n   🔴 From MCD ({len(strategies['mcd'])} variables):")
            for vs in strategies['mcd']:
                scale_info = " [scaled]" if vs.scale_to_era5 else ""
                click.echo(f"      • {vs.name}{scale_info}")
        
        if strategies['era5_mean']:
            click.echo(f"\n   🔵 ERA5 Climatological Mean ({len(strategies['era5_mean'])} variables):")
            for vs in strategies['era5_mean']:
                click.echo(f"      • {vs.name}")
        
        if strategies['constant']:
            click.echo(f"\n   🟢 Constants ({len(strategies['constant'])} variables):")
            for vs in strategies['constant']:
                click.echo(f"      • {vs.name} = {vs.constant_value}")
        
        if strategies['keep_era5']:
            click.echo(f"\n   ⚪ Keep Original ERA5 ({len(strategies['keep_era5'])} variables):")
            for vs in strategies['keep_era5']:
                click.echo(f"      • {vs.name}")
        
        click.echo()
        
        # Expected output
        files_per_day = 24 // formatter_config.time_step_hours
        total_files = formatter_config.num_days * files_per_day
        
        click.echo("📈 Expected Output:")
        click.echo(f"   Files per day:     {files_per_day}")
        click.echo(f"   Total files:       {total_files}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@format_group.command(name='validate')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to formatter config file'
)
def format_validate(config: Path):
    """
    Validate a formatting configuration (check paths, variable names, etc.).
    
    Example:
        graphcast-mars format validate --config format_config.yaml
    """
    try:
        click.echo("🔍 Validating configuration...")
        click.echo()
        
        formatter_config = GraphCastFormatterConfig.from_yaml(config)
        
        errors = []
        warnings = []
        
        # Check paths exist
        if not formatter_config.mcd_data_path.exists():
            errors.append(f"MCD data path does not exist: {formatter_config.mcd_data_path}")
        else:
            click.echo(f"✅ MCD data path exists")
        
        if not formatter_config.era5_sample_path.exists():
            errors.append(f"ERA5 sample path does not exist: {formatter_config.era5_sample_path}")
        else:
            click.echo(f"✅ ERA5 sample path exists")
        
        if not formatter_config.era5_stats_path.exists():
            warnings.append(f"ERA5 stats file not found: {formatter_config.era5_stats_path}")
        else:
            click.echo(f"✅ ERA5 stats file exists")
        
        # Check variable strategies
        mcd_vars = [vs for vs in formatter_config.variable_strategies if vs.strategy == 'mcd']
        if not mcd_vars:
            warnings.append("No MCD variables configured - all data will be from ERA5 or constants")
        else:
            click.echo(f"✅ {len(mcd_vars)} MCD variables configured")
        
        # Check constant values
        for vs in formatter_config.variable_strategies:
            if vs.strategy == 'constant' and vs.constant_value is None:
                errors.append(f"Variable '{vs.name}' has 'constant' strategy but no value")
        
        click.echo()
        
        if errors:
            click.echo("❌ Validation FAILED:")
            for err in errors:
                click.echo(f"   • {err}")
            raise click.Abort()
        
        if warnings:
            click.echo("⚠️  Warnings:")
            for warn in warnings:
                click.echo(f"   • {warn}")
            click.echo()
        
        click.echo("✅ Configuration is valid!")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()