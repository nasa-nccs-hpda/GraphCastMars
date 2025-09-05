#!/bin/bash
# Bash script to run your training script with argparse options

# Path to your Python script
SCRIPT="train.py"   # <-- change this to your actual filename

python3 "$SCRIPT" \
    --apath "../gdata_025_wb" \
    --start-date "1 Jan 2020 00:00" \
    --end-date "31 Dec 2021 18:00" \
    --forecast-length "1" \
    --batch-size 32 \
    --checkpoint-every 10 \
    --learning-rate 1e-6 \
    --debug \
    --dry-run \
    --log-jax-compiles
