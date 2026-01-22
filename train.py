#!/usr/bin/env python
"""Training script wrapper"""

from src.training.trainer import GraphCastTrainer, TrainingConfig

def main():
    config = TrainingConfig.from_yaml("configs/training_config.yaml")
    trainer = GraphCastTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()