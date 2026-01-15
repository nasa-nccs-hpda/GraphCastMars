# src/view/extract_cli.py

import click
from pathlib import Path
from typing import Optional
import logging

from ..preprocessing.mcd_extractor import MCDConfig, MCDExtractor

logger = logging.getLogger(__name__)


@click.group(name='extract')
def extract_group():
    """Commands for extracting MCD data"""
    pass


@extract_group.command(name='run')
@click.option(
    '--data-location',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Path to MCD data directory'
)
@click.option(
    '--output-path',
    type=click.Path(path_type=Path),
    required=True,
    help='Output directory for processed files'
)
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to YAML config file (overrides other options)'
)
@click.option(
    '--ls-start',
    default=0,
    type=int,
    help='Starting solar longitude (Ls) [default: 0]'
)
@click.option(
    '--ls-end',
    default=361,
    type=int,
    help='Ending solar longitude (Ls) [default: 361]'
)
@click.option(
    '--ls-step',
    default=5,
    type=int,
    help='Solar longitude step size [default: 5]'
)
@click.option(
    '--lct-start',
    default=0,
    type=int,
    help='Starting local time (hours) [default: 0]'
)
@click.option(
    '--lct-end',
    default=24,
    type=int,
    help='Ending local time (hours) [default: 24]'
)
@click.option(
    '--lct-step',
    default=6,
    type=int,
    help='Local time step size (hours) [default: 6]'
)
@click.option(
    '--mars-year',
    default=37,
    type=int,
    help='Mars Year for date conversion [default: 37]'
)
@click.option(
    '--skip-existing/--overwrite',
    default=True,
    help='Skip existing files vs overwrite them [default: skip-existing]'
)
@click.option(
    '--data-version',
    default='6.1',
    help='MCD data version [default: 6.1]'
)
@click.option(
    '--zkey',
    default=3,
    type=int,
    help='Vertical coordinate key [default: 3]'
)
@click.option(
    '--hrkey',
    default=0,
    type=int,
    help='High-resolution topography flag (0=off, 1=on) [default: 0]'
)
def extract_run(
    data_location: Path,
    output_path: Path,
    config: Optional[Path],
    ls_start: int,
    ls_end: int,
    ls_step: int,
    lct_start: int,
    lct_end: int,
    lct_step: int,
    mars_year: int,
    skip_existing: bool,
    data_version: str,
    zkey: int,
    hrkey: int
):
    """
    Extract MCD data to GraphCast-ready format.
    
    Example:
        graphcast-mars extract run --data-location /path/to/mcd --output-path ./output
    """
    try:
        # Load config from file if provided, otherwise use CLI args
        if config:
            click.echo(f"Loading configuration from: {config}")
            mcd_config = MCDConfig.from_yaml(config)
        else:
            mcd_config = MCDConfig(
                data_location=data_location,
                output_path=output_path,
                data_version=data_version,
                ls_range=(ls_start, ls_end, ls_step),
                lct_range=(lct_start, lct_end, lct_step),
                mars_year=mars_year,
                zkey=zkey,
                hrkey=hrkey
            )
        
        click.echo(f"📂 Data location: {mcd_config.data_location}")
        click.echo(f"📂 Output path: {mcd_config.output_path}")
        click.echo(f"🌍 Mars Year: {mcd_config.mars_year}")
        click.echo(f"🔄 Ls range: {ls_start}° to {ls_end}° (step: {ls_step}°)")
        click.echo(f"🕐 Local time range: {lct_start}h to {lct_end}h (step: {lct_step}h)")
        click.echo()
        
        # Create extractor and run
        extractor = MCDExtractor(mcd_config)
        
        click.echo("🚀 Starting extraction...")
        output_files = extractor.extract_range(skip_existing=skip_existing)
        
        click.echo()
        click.echo(f"✅ Extraction complete!")
        click.echo(f"📊 Generated {len(output_files)} files")
        click.echo(f"📂 Output directory: {mcd_config.output_path.absolute()}")
        
    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        logger.exception("Error during extraction")
        raise click.Abort()


@extract_group.command(name='single')
@click.option(
    '--data-location',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Path to MCD data directory'
)
@click.option(
    '--output-path',
    type=click.Path(path_type=Path),
    required=True,
    help='Output directory for processed file'
)
@click.option(
    '--ls',
    required=True,
    type=float,
    help='Solar longitude (Ls) [0-360 degrees]'
)
@click.option(
    '--lct',
    required=True,
    type=float,
    help='Local time in hours [0-24]'
)
@click.option(
    '--output-file',
    type=click.Path(path_type=Path),
    help='Custom output filename (optional)'
)
@click.option(
    '--mars-year',
    default=37,
    type=int,
    help='Mars Year for date conversion [default: 37]'
)
def extract_single(
    data_location: Path,
    output_path: Path,
    ls: float,
    lct: float,
    output_file: Optional[Path],
    mars_year: int
):
    """
    Extract MCD data for a single time point.
    
    Example:
        graphcast-mars extract single --data-location /path/to/mcd --output-path ./output --ls 90 --lct 12
    """
    try:
        config = MCDConfig(
            data_location=data_location,
            output_path=output_path,
            mars_year=mars_year
        )
        
        click.echo(f"📍 Extracting single point: Ls={ls}°, Local time={lct}h")
        
        extractor = MCDExtractor(config)
        result_file = extractor.extract_single(ls, lct, output_file)
        
        click.echo(f"✅ Saved to: {result_file.absolute()}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("Error during single extraction")
        raise click.Abort()


@extract_group.command(name='generate-config')
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    required=True,
    help='Path for config template file'
)
@click.option(
    '--example',
    type=click.Choice(['minimal', 'full']),
    default='full',
    help='Template type: minimal or full [default: full]'
)
def extract_generate_config(output: Path, example: str):
    """
    Generate a template configuration file for MCD extraction.
    
    Example:
        graphcast-mars extract generate-config --output my_config.yaml --example full
    """
    try:
        config = MCDConfig(
            data_location='/path/to/mcd/data',
            output_path='./output',
            data_version='6.1',
            mars_year=37
        )
        
        if example == 'minimal':
            # Only save essential parameters
            pass  # You could create a minimal version
        
        config.to_yaml(output)
        click.echo(f"✅ Config template saved to: {output.absolute()}")
        click.echo(f"📝 Edit the file and use with: --config {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@extract_group.command(name='info')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to config file'
)
def extract_info(config: Path):
    """
    Display information about an extraction configuration.
    
    Example:
        graphcast-mars extract info --config my_config.yaml
    """
    try:
        mcd_config = MCDConfig.from_yaml(config)
        
        click.echo("📋 Configuration Summary")
        click.echo("=" * 50)
        click.echo(f"Data location:     {mcd_config.data_location}")
        click.echo(f"Output path:       {mcd_config.output_path}")
        click.echo(f"Data version:      {mcd_config.data_version}")
        click.echo(f"Mars Year:         {mcd_config.mars_year}")
        click.echo(f"Ls range:          {mcd_config.ls_range[0]}° to {mcd_config.ls_range[1]}° (step: {mcd_config.ls_range[2]}°)")
        click.echo(f"Local time range:  {mcd_config.lct_range[0]}h to {mcd_config.lct_range[1]}h (step: {mcd_config.lct_range[2]}h)")
        click.echo(f"Vertical key:      {mcd_config.zkey}")
        click.echo(f"Hi-res topo:       {'Yes' if mcd_config.hrkey else 'No'}")
        click.echo()
        
        # Calculate expected number of files
        ls_steps = len(range(*mcd_config.ls_range))
        lct_steps = len(range(*mcd_config.lct_range))
        total_files = ls_steps * lct_steps
        
        click.echo(f"📊 Expected output: {total_files} files")
        click.echo(f"   ({ls_steps} Ls steps × {lct_steps} time steps)")
        
        # Check if 2D/3D variables configured
        click.echo()
        click.echo(f"2D variables:      {len(mcd_config.variables_2d)}")
        click.echo(f"3D variables:      {len(mcd_config.variables_3d)}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()