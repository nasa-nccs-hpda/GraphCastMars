# src/view/cli.py

import click
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    GraphCast Mars - Climate Data Processing Pipeline
    
    Process Mars Climate Database (MCD) data and format for GraphCast predictions.
    """
    pass


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()