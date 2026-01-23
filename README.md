# GraphCast Mars - Fine-tuning GraphCast for Mars Climate Prediction

Fine-tune DeepMind's GraphCast weather prediction model on Mars Climate Database (MCD) data to predict Martian temperature variations.

## Quick Start (No Installation Required)

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Access to MCD data files
- GraphCast checkpoint and normalization statistics

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/graphcast-mars.git
cd graphcast-mars
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup GraphCast Source Code

```bash
# Clone GraphCast into external/
bash scripts/setup_dev_env.sh
```

Or manually:
```bash
mkdir -p external
cd external
git clone https://github.com/google-deepmind/graphcast.git
cd ..

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"
```

### 4. Prepare Directory Structure

```bash
# Create necessary directories
mkdir -p data/mcd_raw
mkdir -p data/processed
mkdir -p checkpoints/graphcast
mkdir -p checkpoints/mars
mkdir -p output/training
mkdir -p configs
```

### 5. Download Required Files

Place the following files in `checkpoints/graphcast/`:
- `params_GraphCast_small.npz` - Pre-trained GraphCast checkpoint
- `stats_mean_by_level.nc` - Normalization mean statistics
- `stats_stddev_by_level.nc` - Normalization std deviation statistics
- `stats_diffs_stddev_by_level.nc` - Normalization diff std deviation statistics

### 6. Run MCD Data Extraction

```bash
# Generate extraction config
python -c "
from src.preprocessing.mcd_extractor import MCDConfig
config = MCDConfig(
    data_location='/path/to/your/mcd/data',
    output_path='./data/processed/extracted',
    ls_range=(0, 361, 5),
    lct_range=(0, 24, 6)
)
config.to_yaml('configs/mcd_extraction.yaml')
print('Config saved to configs/mcd_extraction.yaml')
"

# Edit configs/mcd_extraction.yaml with your paths

# Run extraction
python -c "
import sys
sys.path.insert(0, '.')
from src.preprocessing.mcd_extractor import MCDConfig, MCDExtractor

config = MCDConfig.from_yaml('configs/mcd_extraction.yaml')
extractor = MCDExtractor(config)
extractor.extract_range()
print('Extraction complete!')
"
```

### 7. Format Data for GraphCast

```bash
# Generate format config
python -c "
from src.preprocessing.graphcast_formatter import GraphCastFormatterConfig
config = GraphCastFormatterConfig(
    mcd_data_path='./data/processed/extracted',
    era5_sample_path='/path/to/era5/samples',
    era5_stats_path='./checkpoints/graphcast/stats_mean_by_level.nc',
    output_path='./data/processed/formatted'
)
config.to_yaml('configs/graphcast_format.yaml')
print('Config saved to configs/graphcast_format.yaml')
"

# Edit configs/graphcast_format.yaml

# Run formatting
python -c "
import sys
sys.path.insert(0, '.')
from src.preprocessing.graphcast_formatter import GraphCastFormatterConfig, GraphCastFormatter

config = GraphCastFormatterConfig.from_yaml('configs/graphcast_format.yaml')
formatter = GraphCastFormatter(config)
formatter.process_all_dates()
print('Formatting complete!')
"
```

### 8. Train GraphCast Model

**Create training config:**

```bash
cat > configs/training_config.yaml << EOF
checkpoint_path: ./checkpoints/graphcast/params_GraphCast_small.npz
stats_dir: ./checkpoints/graphcast
data_dir: ./data/processed/formatted
output_dir: ./output/training

num_epochs: 100
batch_size: 1
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip: 5.0
save_every: 10
target_lead_times: "6h"
EOF
```

**Run training:**

```bash
# Make sure PYTHONPATH includes external/graphcast
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"

# Run training script
python train.py --config configs/training_config.yaml
```

### 9. Monitor Training

Training outputs will be saved to `output/training/`:
- `checkpoint_step_XXXXX.npz` - Periodic checkpoints
- `best_model.npz` - Best model based on training loss

Watch the console for progress:
```
2024-01-22 10:00:00 - INFO - Epoch 1/100
2024-01-22 10:00:01 - INFO - Step 0, Loss: 0.523142
2024-01-22 10:00:02 - INFO - Step 10, Loss: 0.487253
...
```

---

## Alternative: Simple One-File Training Script

If you prefer a single script without the modular structure:

**Create `simple_train.py`:**

```python
#!/usr/bin/env python
"""Simple training script - no package installation needed"""

import sys
import os
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external" / "graphcast"))

# Remove LD_PRELOAD warning
os.environ.pop('LD_PRELOAD', None)

from src.training.trainer import GraphCastTrainer, TrainingConfig
import logging

logging.basicConfig(level=logging.INFO)

# Configure training
config = TrainingConfig(
    checkpoint_path="./checkpoints/graphcast/params_GraphCast_small.npz",
    stats_dir="./checkpoints/graphcast",
    data_dir="./data/processed/formatted",
    output_dir="./output/training",
    num_epochs=100,
    batch_size=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    gradient_clip=5.0,
    save_every=10,
    target_lead_times="6h"
)

# Train
trainer = GraphCastTrainer(config)
trainer.train()
```

**Run it:**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"
python simple_train.py
```

---

## Directory Structure After Setup

```
graphcast-mars/
├── external/
│   └── graphcast/              # GraphCast source (cloned)
├── data/
│   ├── mcd_raw/                # Your MCD data
│   ├── processed/
│   │   ├── extracted/          # Extracted MCD files
│   │   └── formatted/          # GraphCast-ready files
├── checkpoints/
│   ├── graphcast/              # Pre-trained checkpoint & stats
│   └── mars/                   # Your fine-tuned models
├── output/
│   └── training/               # Training outputs
├── configs/
│   ├── mcd_extraction.yaml
│   ├── graphcast_format.yaml
│   └── training_config.yaml
├── src/
│   ├── preprocessing/
│   ├── training/
│   └── view/
├── train.py                    # Main training script
├── simple_train.py             # Simple one-file version
└── requirements.txt
```

---

## Troubleshooting

### PYTHONPATH Issues

If you see `ImportError: No module named 'graphcast'`:

```bash
# Set PYTHONPATH every time
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"

# Or create activation script
echo 'export PYTHONPATH="${PYTHONPATH}:'$(pwd)'/external/graphcast"' > activate.sh
source activate.sh
```

### JAX/CUDA Issues

```bash
# Check JAX can see GPU
python -c "import jax; print(jax.devices())"

# Should output: [cuda(id=0), ...]
```

If no GPU found, reinstall JAX with CUDA support:
```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Memory Issues

Reduce batch size or use gradient accumulation:
```yaml
batch_size: 1  # Already minimal
gradient_clip: 5.0  # Helps prevent memory spikes
```

### Data Loading Errors

Check your data files:
```bash
python -c "
import xarray as xr
ds = xr.open_dataset('data/processed/formatted/your_file.nc')
print('Time steps:', ds.sizes['time'])
print('Variables:', list(ds.data_vars))
"
```

Should show 3 timesteps and all required variables.

---

## Quick Reference Commands

```bash
# Setup
git clone <repo>
cd graphcast-mars
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_dev_env.sh

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/graphcast"

# Train
python train.py --config configs/training_config.yaml

# Or simple version
python simple_train.py
```

---

## Next Steps

- **Evaluate predictions**: Use trained model for inference
- **Visualize results**: Plot temperature predictions vs. actual
- **Experiment**: Try different hyperparameters in config
- **Scale up**: Add more MCD variables beyond temperature

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/graphcast-mars/issues)
- **GraphCast Documentation**: [DeepMind GraphCast](https://github.com/google-deepmind/graphcast)
- **MCD Documentation**: [Mars Climate Database](http://www-mars.lmd.jussieu.fr/mcd_python/)

---

## License

Apache 2.0 (same as GraphCast)