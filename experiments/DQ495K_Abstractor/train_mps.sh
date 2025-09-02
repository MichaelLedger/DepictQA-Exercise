#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH

# Run training with MPS backend
python3 $src_dir/train_mps.py --cfg config_mps.yaml --batch_size 4
