# src/view/__init__.py

from .cli import cli
from .extract_cli import extract_group
from .format_cli import format_group
from .train_cli import train_group

# Register command groups
cli.add_command(extract_group)
cli.add_command(format_group)
cli.add_command(train_group)

__all__ = ['cli']