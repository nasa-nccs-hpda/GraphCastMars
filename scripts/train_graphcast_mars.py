# scripts/train_graphcast_mars.py
# Backward compatibility wrapper

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import GraphCastTrainer, TrainingConfig

def main():
    config = TrainingConfig(
        checkpoint_path="/path/to/params_GraphCast_small.npz",
        stats_dir="/path/to/stats",
        data_dir="/path/to/mcd_data",
        output_dir="./output/training",
        num_epochs=100,
        learning_rate=1e-4
    )
    
    trainer = GraphCastTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()