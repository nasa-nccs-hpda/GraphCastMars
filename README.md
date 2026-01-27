```markdown
# GraphCast Mars - Climate Data Processing and Prediction Pipeline

A production-ready pipeline for processing Mars Climate Database (MCD) data and running GraphCast predictions for Mars temperature forecasting.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This package provides a complete workflow for:
1. **Extracting MCD data** to GraphCast-ready format
2. **Formatting data** by combining MCD variables with ERA5 structure
3. **Running predictions** using pre-trained or fine-tuned GraphCast models
4. *(Optional)* **Training/fine-tuning** GraphCast models on Mars data

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/graphcast-mars.git
cd graphcast-mars

# Create virtual environment (recommended)
conda create -n graphcast-mars python=3.10
conda activate graphcast-mars

# Install package
pip install -e .

# Verify installation
graphcast-mars --help
```

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for inference/training)
- ~50GB free disk space
- MCD data files
- Pre-trained GraphCast checkpoint

### Directory Setup

```bash
# Create required directories
mkdir -p data/{mcd_raw,mcd_processed,graphcast_ready}
mkdir -p models/checkpoints
mkdir -p predictions
mkdir -p configs
```

## Basic Workflow

### Step 1: Extract MCD Data

Extract Mars Climate Database variables to NetCDF format.

```bash
# Generate configuration template
graphcast-mars extract generate-config --output configs/extract.yaml

# Edit configs/extract.yaml with your paths, then run
graphcast-mars extract run --config configs/extract.yaml
```

**Example config** (`configs/extract.yaml`):
```yaml
data_location: /path/to/mcd/data
output_path: ./data/mcd_processed
data_version: '6.1'

ls_range: [0, 361, 5]        # Solar longitude: start, stop, step
lct_range: [0, 24, 6]         # Local time: start, stop, step
mars_year: 37

zkey: 3                       # Vertical coordinate (3=height above surface)
hrkey: 0                      # High-res topography (0=off)
```

**Quick CLI usage** (without config file):
```bash
graphcast-mars extract run \
  --data-location /path/to/mcd/data \
  --output-path ./data/mcd_processed \
  --ls-start 0 --ls-end 360 --ls-step 10 \
  --lct-start 0 --lct-end 24 --lct-step 6
```

### Step 2: Format for GraphCast

Combine MCD variables with ERA5 structure to create GraphCast-ready input files.

```bash
# Generate configuration template
graphcast-mars format generate-config \
  --output configs/format.yaml \
  --template temperature-only

# Edit configs/format.yaml, then run
graphcast-mars format run --config configs/format.yaml
```

**Example config** (`configs/format.yaml`):
```yaml
# Input paths
mcd_data_path: ./data/mcd_processed
era5_sample_path: /path/to/era5/samples
era5_stats_path: /path/to/era5/stats/stats_mean_by_level.nc
output_path: ./data/graphcast_ready

# Date range
start_date: "2022-03-20"
num_days: 361

# Time settings
time_step_hours: 6
num_input_steps: 2
num_output_steps: 1

# Spatial
target_resolution: 1.0

# Variable strategies
variable_strategies:
  # Use MCD temperature (scaled to ERA5 range)
  - name: '2m_temperature'
    strategy: 'mcd'
    scale_to_era5: true
  
  - name: 'temperature'
    strategy: 'mcd'
    scale_to_era5: true
  
  # Use MCD topography
  - name: 'geopotential_at_surface'
    strategy: 'mcd'
    scale_to_era5: false
  
  # Set to ERA5 climatological mean
  - name: 'geopotential'
    strategy: 'era5_mean'
  
  - name: 'mean_sea_level_pressure'
    strategy: 'era5_mean'
  
  # ... (other variables)
  
  # Set to constants
  - name: 'land_sea_mask'
    strategy: 'constant'
    constant_value: 1.0
  
  - name: 'total_precipitation_6hr'
    strategy: 'constant'
    constant_value: 0.0
```

**Validate configuration**:
```bash
graphcast-mars format validate --config configs/format.yaml
```

**Process single date** (for testing):
```bash
graphcast-mars format run --config configs/format.yaml --date 2022-03-20
```

### Step 3: Run Predictions

Use pre-trained GraphCast-Mars model to generate predictions.

```bash
# Generate inference configuration
graphcast-mars predict generate-config --output configs/inference.yaml

# Edit configs/inference.yaml, then run
graphcast-mars predict run --config configs/inference.yaml
```

**Example config** (`configs/inference.yaml`):
```yaml
# Model checkpoint
model_checkpoint: /path/to/params_GraphCast_small.npz

# Normalization statistics
stats_dir: /path/to/stats/

# Input/output
input_data_path: ./data/graphcast_ready/initial_conditions.nc
output_path: ./predictions

# Prediction settings
num_steps: 10                 # Number of forecast steps
lead_time_hours: 6            # Hours per step (total: 10 * 6 = 60 hours)
autoregressive: true          # Use previous prediction as next input
use_chunked_prediction: true  # Memory-efficient rollout

# Output
save_format: netcdf
save_intermediate: false      # Save each timestep separately
compress: true
```

**Quick prediction** (without config):
```bash
graphcast-mars predict run \
  --config configs/inference.yaml \
  --num-steps 20 \
  --visualize
```

## Command Reference

### Extract Commands

```bash
# Run extraction
graphcast-mars extract run --config CONFIG

# Extract single time point
graphcast-mars extract single \
  --data-location /path/to/mcd \
  --output-path ./output \
  --ls 90 --lct 12

# Show config info
graphcast-mars extract info --config CONFIG

# Generate config template
graphcast-mars extract generate-config --output CONFIG
```

### Format Commands

```bash
# Run formatting
graphcast-mars format run --config CONFIG

# Process single date
graphcast-mars format run --config CONFIG --date 2022-03-20

# Validate configuration
graphcast-mars format validate --config CONFIG

# Show config info
graphcast-mars format info --config CONFIG

# Generate config template
graphcast-mars format generate-config --output CONFIG --template [temperature-only|multi-variable]
```

### Predict Commands

```bash
# Run prediction
graphcast-mars predict run --config CONFIG

# Override settings
graphcast-mars predict run --config CONFIG --num-steps 20 --visualize

# Generate config template
graphcast-mars predict generate-config --output CONFIG
```

## Example: Complete Pipeline

```bash
# 1. Extract MCD data
graphcast-mars extract run \
  --data-location /discover/nobackup/mcd_data \
  --output-path ./data/mcd_processed \
  --ls-start 0 --ls-end 360 --ls-step 5

# 2. Format for GraphCast
graphcast-mars format run --config configs/format_mars_temp.yaml

# 3. Run predictions
graphcast-mars predict run \
  --config configs/inference.yaml \
  --num-steps 10 \
  --visualize
```

## Output Files

### After Extraction
```
data/mcd_processed/
├── mcd_output_Ls000_hr00.nc
├── mcd_output_Ls000_hr06.nc
├── ...
└── mcd_output_Ls360_hr18.nc
```

### After Formatting
```
data/graphcast_ready/
├── graphcast_dataset_source-era5-mcd_date-2022-03-20-T00_res-1.0_levels-13_steps-3.nc
├── graphcast_dataset_source-era5-mcd_date-2022-03-20-T06_res-1.0_levels-13_steps-3.nc
├── ...
```

### After Prediction
```
predictions/
├── predictions.nc              # All timesteps combined
├── prediction_step_000.nc      # Individual steps (if save_intermediate=true)
├── prediction_step_001.nc
└── ...
```

## Configuration Details

### Variable Strategies

When formatting data, you can specify how each variable is handled:

- **`mcd`**: Use MCD data (optionally scaled to ERA5 range)
- **`era5_mean`**: Use ERA5 climatological mean
- **`constant`**: Set to a constant value
- **`keep_era5`**: Keep original ERA5 data

**Example: Add more MCD variables**
```yaml
variable_strategies:
  # MCD variables
  - name: '2m_temperature'
    strategy: 'mcd'
    scale_to_era5: true
  
  - name: '10m_u_component_of_wind'
    strategy: 'mcd'              # Now using MCD wind
    scale_to_era5: true
  
  - name: '10m_v_component_of_wind'
    strategy: 'mcd'
    scale_to_era5: true
```

## Troubleshooting

### Common Issues

**1. ImportError: attempted relative import**
```bash
# Run as module from project root
python -m src.inference.predictor

# Or install package
pip install -e .
```

**2. CUDA out of memory**
```yaml
# In inference config, enable chunked prediction
use_chunked_prediction: true

# Or reduce number of steps
num_steps: 5
```

**3. Missing stats files**
```
Ensure you have:
- stats_mean_by_level.nc
- stats_stddev_by_level.nc
- stats_diffs_stddev_by_level.nc

Place in stats_dir specified in config.
```

**4. Resolution mismatch**
```
Model resolution doesn't match data resolution.
Check that:
- MCD data is regridded to 1.0° (if using 1° model)
- ERA5 samples match model resolution
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
graphcast-mars extract run --config configs/extract.yaml
```

## Advanced Usage

### Python API

```python
from pathlib import Path
from src.inference.predictor import run_inference

# Simple prediction
output = run_inference(
    checkpoint_path=Path("models/params_GraphCast_small.npz"),
    stats_dir=Path("models/stats/"),
    input_data_path=Path("data/graphcast_ready/input.nc"),
    output_path=Path("predictions/"),
    num_steps=10
)
```

### Custom Preprocessing

```python
from src.preprocessing.mcd_extractor import MCDExtractor, MCDConfig

# Create custom config
config = MCDConfig(
    data_location="/path/to/mcd",
    output_path="./output",
    ls_range=(0, 180, 10),  # Custom range
    lct_range=(0, 24, 12)    # Every 12 hours
)

# Run extraction
extractor = MCDExtractor(config)
files = extractor.extract_range()
```

## Optional: Model Training/Fine-tuning

For users who want to fine-tune GraphCast on Mars data:

### Setup Training

```bash
# Prepare training data (formatted MCD + ERA5)
graphcast-mars format run --config configs/format_training.yaml

# Split into train/validation sets (80/20 split happens automatically)
```

### Training Configuration

Create `configs/training.yaml`:
```yaml
# Data
data_dir: ./data/graphcast_ready
stats_dir: ./models/stats
checkpoint_dir: ./models/checkpoints

# Model config
model_config:
  resolution: 1.0
  mesh_size: 5
  latent_size: 512
  gnn_msg_steps: 16

# Training
batch_size: 4
learning_rate: 1e-5
num_epochs: 100
save_every: 10

# Data
train_ratio: 0.8
target_lead_times: "6h"
```

### Run Training

```python
# src/training/train.py
from src.training.trainer import GraphCastTrainer, TrainingConfig

config = TrainingConfig.from_yaml("configs/training.yaml")
trainer = GraphCastTrainer(config)
trainer.train()
```

**Note**: Training requires significant computational resources (GPU with >16GB VRAM recommended).

## Project Structure

```
graphcast-mars/
├── src/
│   ├── preprocessing/
│   │   ├── mcd_extractor.py        # MCD data extraction
│   │   └── graphcast_formatter.py  # Format for GraphCast
│   ├── inference/
│   │   ├── predictor.py            # Run predictions
│   │   └── postprocessing.py       # Visualization
│   ├── models/
│   │   ├── model_builder.py        # Build GraphCast model
│   │   └── checkpoint_utils.py     # Checkpoint management
│   ├── training/                   # (Optional) Training code
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── view/
│       └── cli.py                  # Command-line interface
├── configs/                        # Configuration files
├── data/                          # Data directories
├── models/                        # Model checkpoints
├── predictions/                   # Prediction outputs
└── README.md
```

## Data Requirements

### Input Data

- **MCD Data**: Mars Climate Database files (version 6.1 recommended)
- **ERA5 Samples**: Example ERA5 files for structure reference
- **ERA5 Statistics**: Normalization statistics (mean, stddev)
- **GraphCast Checkpoint**: Pre-trained model weights

### Disk Space

- MCD extracted data: ~10GB (for 1 Mars year)
- Formatted GraphCast data: ~20GB
- Model checkpoints: ~500MB
- Predictions: ~1GB per 10-day forecast

## Performance

- **Extraction**: ~2-5 minutes per Mars day (CPU)
- **Formatting**: ~5-10 minutes per Mars day (CPU)
- **Prediction**: ~1-2 minutes per 10 forecast steps (GPU)

## Citation

If you use this code, please cite:

```bibtex
@software{graphcast_mars,
  title={GraphCast Mars: Climate Data Processing and Prediction Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/graphcast-mars}
}
```

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/graphcast-mars/issues)
- **Documentation**: [Full Documentation](https://your-docs-url.com)
- **Contact**: your.email@example.com

## Acknowledgments

- Google DeepMind for GraphCast
- NASA/JPL for Mars Climate Database
- ECMWF for ERA5 data

---

**Quick Links:**
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Troubleshooting](#troubleshooting)
- [Training Guide](#optional-model-trainingfine-tuning)
```

This README:
- ✅ Focuses on preprocessing and inference as the main workflow
- ✅ Provides clear step-by-step instructions
- ✅ Includes example commands and configurations
- ✅ Has training as an optional section
- ✅ Includes troubleshooting and common issues
- ✅ Production-ready with proper structure