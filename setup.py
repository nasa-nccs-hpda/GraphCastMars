from setuptools import setup, find_packages

setup(
    name="graphcast-mars",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'graphcast-mars=src.view.cli:main',
        ],
    },
    install_requires=[
        'click>=8.0',
        'xarray',
        'numpy',
        'pandas',
        'pyyaml',
        'xesmf',
        'jax',
        'optax',
        'dm-haiku',
    ],
)