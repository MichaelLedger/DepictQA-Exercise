#!/bin/bash
export PYTHONPATH=../../src/:$PYTHONPATH
source ../../venv_py39/bin/activate
python -m serve.controller
