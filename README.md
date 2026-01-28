# GraphCast Mars - Climate Data Processing and Prediction Pipeline

Process Mars Climate Database (MCD) data and run GraphCast predictions for Mars temperature forecasting. Optionally fine-tune GraphCast models on Mars data.

## Quick Start (No Installation Required)

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU support)
- Access to MCD data files
- GraphCast checkpoint and normalization statistics
- ~50GB free disk space


### Step 1: Clone Repository

```bash
git clone https://github.com/nasa-nccs-hpda/GraphCastMars.git
cd GraphCastMars
```

### Step 2: Setup Python Environment

```bash
# Load Python/Anaconda (if on DISCOVER)
module load anaconda
```

**Optioanl : Create virtual environment**
```bash 
# Recommended when not using DISCOVER or another managed environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Setup GraphCast Source Code & MCD Library

```bash
# Run setup script
source scripts/setup_environment.sh
```

**Or manually:**
```bash
mkdir -p external
cd external
git clone https://github.com/google-deepmind/graphcast.git
cd ..

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"
```

### Step 4: Prepare Directory Structure

```bash
# Create necessary directories
mkdir -p data/mcd_processed
mkdir -p data/graphcast_ready
mkdir -p checkpoints/graphcast
mkdir -p predictions
```

### Step 5: Download Required Files

Place the following files in `checkpoints/graphcast/`:

```bash
cd checkpoints/graphcast

# Copy from existing location (example for DISCOVER)
cp /discover/nobackup/jli30/systest/Graphcast_Mars_test/checkpoints/graphcast/* .

# Files needed:
# - params_GraphCast_mars.npz (pre-trained checkpoint)
# - stats_mean_by_level.nc (normalization mean)
# - stats_stddev_by_level.nc (normalization std dev)
# - stats_diffs_stddev_by_level.nc (normalization diff std dev)

cd ../..
```

## Main Workflow: Preprocessing and Prediction

### Step 6: Extract MCD Data
**Update configuration:**
```bash
python -c "
from src.preprocessing.mcd_extractor import MCDConfig

config = MCDConfig(
    output_path='./data/mcd_processed',
    data_version='6.1',
    ls_range=(0, 5, 1),      # Solar longitude: 0-360°, every 5°
    lct_range=(0, 24, 6),       # Local time: 0-24h, every 6h
    zkey=3,                     # Height above surface
    hrkey=0                     # No high-res topography
)
config.to_yaml('configs/mcd_extraction.yaml')
print('√ Config saved to configs/mcd_extraction.yaml')
"
```

**Edit the config manually** if needed:
```bash
nano configs/mcd_extraction.yaml
```

**Run extraction:**
```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.preprocessing.mcd_extractor import MCDConfig, MCDExtractor

config = MCDConfig.from_yaml('configs/mcd_extraction.yaml')
extractor = MCDExtractor(config)
output_files = extractor.extract_range()

print(f'√ Extraction complete! Generated {len(output_files)} files')
print(f'√ Output: {config.output_path}')
"
```

**Expected output:**
```
data/mcd_processed/
├── mcd_output_2022-03-20_hr00.nc
├── mcd_output_2022-03-20_hr06.nc
├── mcd_output_2022-03-20_hr12.nc
└── ...
```

### Step 7: Format Data for GraphCast

**Edit variable strategies** (which variables come from MCD vs constants):
```bash
nano configs/graphcast_format.yaml
```

Key section to customize:
```yaml
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
  
  # Set to constants
  - name: 'land_sea_mask'
    strategy: 'constant'
    constant_value: 1.0
```

**Run formatting:**
```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.preprocessing.graphcast_formatter import GraphCastFormatterConfig, GraphCastFormatter

config = GraphCastFormatterConfig.from_yaml('configs/graphcast_format.yaml')
formatter = GraphCastFormatter(config)
results = formatter.process_all_dates()

total_files = sum(len(v) for v in results.values())
print(f'√ Formatting complete! Generated {total_files} files')
print(f'√ Output: {config.output_path}')
"
```

**Expected output:**
```
data/graphcast_ready/
├── graphcast_dataset_source-era5-mcd_date-2022-03-20-T00_res-1.0_levels-13_steps-10.nc
├── graphcast_dataset_source-era5-mcd_date-2022-03-20-T06_res-1.0_levels-13_steps-10.nc
└── ...
```

### Step 8: Run Predictions


**Run prediction:**
```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.inference.predictor import GraphCastPredictor, InferenceConfig

config = InferenceConfig.from_yaml('configs/inference.yaml')
predictor = GraphCastPredictor(config)

print('Running prediction...')
output_file = predictor.predict_and_save()

print(f'√ Prediction complete!')
print(f'V Output: {output_file}')
"
```

**Expected output:**
```
predictions/
└── predictions.nc  (All 10 forecast steps)
```

### Step 9: Verify Results

```bash
python -c "
import xarray as xr

# Load predictions
ds = xr.open_dataset('predictions/predictions.nc')

print('Prediction Summary:')
print(f'  Time steps: {ds.sizes[\"time\"]}')
print(f'  Variables: {list(ds.data_vars.keys())}')
print(f'  Spatial: {ds.sizes[\"lat\"]}x{ds.sizes[\"lon\"]}')
print(f'  Levels: {ds.sizes.get(\"level\", \"N/A\")}')

# Check temperature range
temp = ds['2m_temperature']
print(f'\n2m Temperature:')
print(f'  Min: {float(temp.min()):.2f} K')
print(f'  Max: {float(temp.max()):.2f} K')
print(f'  Mean: {float(temp.mean()):.2f} K')
"
```

---

## Alternative: Using Command-Line Interface (Coming Soon)

Once installed, you can use CLI commands:

```bash
# Install package
pip install -e .

# Extract MCD data
graphcast-mars extract run --config configs/mcd_extraction.yaml

# Format for GraphCast
graphcast-mars format run --config configs/graphcast_format.yaml

# Run predictions
graphcast-mars predict run --config configs/inference.yaml
```

---

## Optional: Fine-tune GraphCast Model

For users who want to fine-tune GraphCast on Mars data:

### Step 10: Prepare Training Data

Use the formatted data from Step 7 as training input.

### Step 11: Create Training Configuration

```bash
cat > configs/training_config.yaml << EOF
checkpoint_path: ./checkpoints/graphcast/params_GraphCast_small.npz
stats_dir: ./checkpoints/graphcast
data_dir: ./data/graphcast_ready
output_dir: ./checkpoints/mars

num_epochs: 100
batch_size: 1
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip: 5.0
save_every: 10
target_lead_times: "6h"

# Data split
train_ratio: 0.8
shuffle: true
EOF
```

### Step 12: Run Training

```bash
# Ensure PYTHONPATH includes GraphCast
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"

# Run training
python train.py --config configs/training_config.yaml
```

### Step 13: Monitor Training

Training outputs in `checkpoints/mars/`:
```
checkpoints/mars/
├── checkpoint_epoch_010.npz
├── checkpoint_epoch_020.npz
├── ...
└── checkpoint_final.npz
```

Console output:
```
2024-01-22 10:00:00 - INFO - Epoch 1/100
2024-01-22 10:00:01 - INFO - Step 0, Loss: 0.523142
2024-01-22 10:00:02 - INFO - Step 10, Loss: 0.487253
...
2024-01-22 10:15:30 - INFO - ✓ Saved checkpoint: checkpoint_epoch_010.npz
```

### Step 14: Use Fine-tuned Model for Predictions

Update `configs/inference.yaml`:
```yaml
model_checkpoint: ./checkpoints/mars/checkpoint_final.npz
```

Then run predictions as in Step 8.

---

## Troubleshooting

### Common Issues

**1. Module import errors**
```bash
# Ensure GraphCast is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"

# Or add at start of script
import sys
sys.path.insert(0, './external/graphcast')
```

**2. CUDA out of memory**
```yaml
# In inference.yaml
use_chunked_prediction: true
num_steps: 5  # Reduce number of steps
```

**3. Missing normalization statistics**
```
Ensure all three stats files are in checkpoints/graphcast/:
- stats_mean_by_level.nc
- stats_stddev_by_level.nc
- stats_diffs_stddev_by_level.nc
```

**4. Data format errors**
```bash
# Verify MCD extraction output
python -c "
import xarray as xr
ds = xr.open_dataset('data/mcd_processed/mcd_output_Ls000_hr00.nc')
print(ds)
"
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with detailed output
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# ... your code ...
"
```

---

## Project Structure

```
GraphCastMars/
├── src/
│   ├── preprocessing/
│   │   ├── mcd_extractor.py        # Extract MCD data
│   │   └── graphcast_formatter.py  # Format for GraphCast
│   ├── inference/
│   │   ├── predictor.py            # Run predictions
│   │   └── postprocessing.py       # Visualization
│   ├── models/
│   │   ├── model_builder.py        # Build GraphCast model
│   │   └── checkpoint_utils.py     # Checkpoint management
│   ├── training/                   # (Optional) Training
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── view/
│       └── cli.py                  # CLI (coming soon)
├── configs/                        # Configuration files
├── data/
│   ├── mcd_raw/                   # Raw MCD data
│   ├── mcd_processed/             # Extracted NetCDF
│   └── graphcast_ready/           # Formatted for GraphCast
├── checkpoints/
│   ├── graphcast/                 # Pre-trained model
│   └── mars/                      # Fine-tuned model
├── predictions/                   # Prediction outputs
├── train.py                       # Training script
├── requirements.txt
└── README.md
```

---

## Requirements

### Computational Resources

- **Preprocessing**: CPU, 8GB RAM
- **Inference**: GPU (8GB+ VRAM), 16GB RAM
- **Training**: GPU (16GB+ VRAM), 32GB RAM

### Disk Space

- MCD raw data: ~5GB
- MCD processed: ~10GB
- Formatted data: ~20GB
- Checkpoints: ~500MB each
- Predictions: ~1GB per 10-day forecast

### Performance Estimates

- **MCD Extraction**: ~2-5 min/Mars day (CPU)
- **Formatting**: ~5-10 min/Mars day (CPU)
- **Inference**: ~1-2 min per 10 steps (GPU)
- **Training**: ~10-20 min/epoch (GPU)

---

## Citation

If you use this code, please cite:

```bibtex
@software{graphcast_mars,
  title={GraphCast Mars: Climate Data Processing and Prediction Pipeline},
  author={NASA NCCS},
  year={2024},
  url={https://github.com/nasa-nccs-hpda/GraphCastMars}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Google DeepMind for GraphCast
- NASA/JPL for Mars Climate Database
- ECMWF for ERA5 data

---

**Need Help?**
- [Issues](https://github.com/nasa-nccs-hpda/GraphCastMars/issues)
- [Documentation](https://your-docs-url.com)
- Contact: your.email@nasa.gov
```

This version:
- ✅ Clear step-by-step workflow with numbered steps
- ✅ Focuses on preprocessing → prediction as main workflow
- ✅ Training is clearly marked as optional
- ✅ Python snippets can be run directly (no installation needed initially)
- ✅ Shows expected outputs at each step
- ✅ Includes verification steps
- ✅ Comprehensive troubleshooting section
- ✅ Matches your preferred structure and clarity
