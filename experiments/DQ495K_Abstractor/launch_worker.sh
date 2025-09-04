#!/bin/bash
export PYTHONPATH=../../src/:$PYTHONPATH

# Run worker with MPS configuration
python -m serve.depictqa_worker --cfg config_mps.yaml --cfg_serve serve.yaml