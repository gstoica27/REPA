#!/bin/bash

# 1.25 - 2.25
# 0.5 - 1.0
# 75 - 150

args="--beaker-image ai2/cuda11.8-cudnn8-dev-ubuntu20.04 --weka prior-default:/weka/prior-default --weka oe-training-default:/weka/oe-training-default --budget ai2/prior --shared-memory 800G --priority high --allow-dirty"

: '
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.5-75 -- bash eval_script_with_cfg.sh 1.25 0.5 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.5-100 -- bash eval_script_with_cfg.sh 1.25 0.5 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.5-125 -- bash eval_script_with_cfg.sh 1.25 0.5 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.5-150 -- bash eval_script_with_cfg.sh 1.25 0.5 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.5-250 -- bash eval_script_with_cfg.sh 1.25 0.5 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.75-75 -- bash eval_script_with_cfg.sh 1.25 0.75 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.75-100 -- bash eval_script_with_cfg.sh 1.25 0.75 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.75-125 -- bash eval_script_with_cfg.sh 1.25 0.75 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.75-150 -- bash eval_script_with_cfg.sh 1.25 0.75 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-0.75-250 -- bash eval_script_with_cfg.sh 1.25 0.75 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-1.0-75 -- bash eval_script_with_cfg.sh 1.25 1.0 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-1.0-100 -- bash eval_script_with_cfg.sh 1.25 1.0 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-1.0-125 -- bash eval_script_with_cfg.sh 1.25 1.0 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-1.0-150 -- bash eval_script_with_cfg.sh 1.25 1.0 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.25-1.0-250 -- bash eval_script_with_cfg.sh 1.25 1.0 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.5-75 -- bash eval_script_with_cfg.sh 1.75 0.5 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.5-100 -- bash eval_script_with_cfg.sh 1.75 0.5 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.5-125 -- bash eval_script_with_cfg.sh 1.75 0.5 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.5-150 -- bash eval_script_with_cfg.sh 1.75 0.5 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.5-250 -- bash eval_script_with_cfg.sh 1.75 0.5 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-75 -- bash eval_script_with_cfg.sh 1.75 0.75 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-100 -- bash eval_script_with_cfg.sh 1.75 0.75 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-125 -- bash eval_script_with_cfg.sh 1.75 0.75 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-150 -- bash eval_script_with_cfg.sh 1.75 0.75 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-250 -- bash eval_script_with_cfg.sh 1.75 0.75 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-1.0-75 -- bash eval_script_with_cfg.sh 1.75 1.0 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-1.0-100 -- bash eval_script_with_cfg.sh 1.75 1.0 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-1.0-125 -- bash eval_script_with_cfg.sh 1.75 1.0 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-1.0-150 -- bash eval_script_with_cfg.sh 1.75 1.0 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-1.0-250 -- bash eval_script_with_cfg.sh 1.75 1.0 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.5-75 -- bash eval_script_with_cfg.sh 2.25 0.5 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.5-100 -- bash eval_script_with_cfg.sh 2.25 0.5 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.5-125 -- bash eval_script_with_cfg.sh 2.25 0.5 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.5-150 -- bash eval_script_with_cfg.sh 2.25 0.5 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.5-250 -- bash eval_script_with_cfg.sh 2.25 0.5 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.75-75 -- bash eval_script_with_cfg.sh 2.25 0.75 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.75-100 -- bash eval_script_with_cfg.sh 2.25 0.75 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.75-125 -- bash eval_script_with_cfg.sh 2.25 0.75 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.75-150 -- bash eval_script_with_cfg.sh 2.25 0.75 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-0.75-250 -- bash eval_script_with_cfg.sh 2.25 0.75 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-1.0-75 -- bash eval_script_with_cfg.sh 2.25 1.0 75
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-1.0-100 -- bash eval_script_with_cfg.sh 2.25 1.0 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-1.0-125 -- bash eval_script_with_cfg.sh 2.25 1.0 125
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-1.0-150 -- bash eval_script_with_cfg.sh 2.25 1.0 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-2.25-1.0-250 -- bash eval_script_with_cfg.sh 2.25 1.0 250
'

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.65-50 -- bash eval_script_with_cfg.sh 1.75 0.65 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.65-100 -- bash eval_script_with_cfg.sh 1.75 0.65 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.65-150 -- bash eval_script_with_cfg.sh 1.75 0.65 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.65-200 -- bash eval_script_with_cfg.sh 1.75 0.65 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.65-250 -- bash eval_script_with_cfg.sh 1.75 0.65 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.7-50 -- bash eval_script_with_cfg.sh 1.75 0.7 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.7-100 -- bash eval_script_with_cfg.sh 1.75 0.7 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.7-150 -- bash eval_script_with_cfg.sh 1.75 0.7 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.7-200 -- bash eval_script_with_cfg.sh 1.75 0.7 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.7-250 -- bash eval_script_with_cfg.sh 1.75 0.7 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-50 -- bash eval_script_with_cfg.sh 1.75 0.75 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-100 -- bash eval_script_with_cfg.sh 1.75 0.75 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-150 -- bash eval_script_with_cfg.sh 1.75 0.75 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-200 -- bash eval_script_with_cfg.sh 1.75 0.75 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.75-0.75-250 -- bash eval_script_with_cfg.sh 1.75 0.75 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.65-50 -- bash eval_script_with_cfg.sh 1.8 0.65 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.65-100 -- bash eval_script_with_cfg.sh 1.8 0.65 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.65-150 -- bash eval_script_with_cfg.sh 1.8 0.65 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.65-200 -- bash eval_script_with_cfg.sh 1.8 0.65 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.65-250 -- bash eval_script_with_cfg.sh 1.8 0.65 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.7-50 -- bash eval_script_with_cfg.sh 1.8 0.7 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.7-100 -- bash eval_script_with_cfg.sh 1.8 0.7 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.7-150 -- bash eval_script_with_cfg.sh 1.8 0.7 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.7-200 -- bash eval_script_with_cfg.sh 1.8 0.7 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.7-250 -- bash eval_script_with_cfg.sh 1.8 0.7 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.75-50 -- bash eval_script_with_cfg.sh 1.8 0.75 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.75-100 -- bash eval_script_with_cfg.sh 1.8 0.75 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.75-150 -- bash eval_script_with_cfg.sh 1.8 0.75 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.75-200 -- bash eval_script_with_cfg.sh 1.8 0.75 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.8-0.75-250 -- bash eval_script_with_cfg.sh 1.8 0.75 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.65-50 -- bash eval_script_with_cfg.sh 1.85 0.65 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.65-100 -- bash eval_script_with_cfg.sh 1.85 0.65 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.65-150 -- bash eval_script_with_cfg.sh 1.85 0.65 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.65-200 -- bash eval_script_with_cfg.sh 1.85 0.65 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.65-250 -- bash eval_script_with_cfg.sh 1.85 0.65 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.7-50 -- bash eval_script_with_cfg.sh 1.85 0.7 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.7-100 -- bash eval_script_with_cfg.sh 1.85 0.7 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.7-150 -- bash eval_script_with_cfg.sh 1.85 0.7 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.7-200 -- bash eval_script_with_cfg.sh 1.85 0.7 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.7-250 -- bash eval_script_with_cfg.sh 1.85 0.7 250

uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.75-50 -- bash eval_script_with_cfg.sh 1.85 0.75 50
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.75-100 -- bash eval_script_with_cfg.sh 1.85 0.75 100
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.75-150 -- bash eval_script_with_cfg.sh 1.85 0.75 150
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.75-200 -- bash eval_script_with_cfg.sh 1.85 0.75 200
uv run gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --gpus 8 --name cfg-1.85-0.75-250 -- bash eval_script_with_cfg.sh 1.85 0.75 250
