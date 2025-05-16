#!/bin/bash

# 1.25 - 2.25
# 0.5 - 1.0
# 75 - 150

args="--beaker-image ai2/cuda11.8-cudnn8-dev-ubuntu20.04 --weka prior-default:/weka/prior-default --weka oe-training-default:/weka/oe-training-default --budget ai2/prior --shared-memory 800G --priority high --allow-dirty --retries 3"

gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --cluster ai2/saturn-cirrascale --gpus 8 --name SiTXL2-StackedCFG-Ours-cfg-1.85-0.65-50 -- bash eval_script_with_cfg_stacked_ours.sh 1.85 0.65 50
gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --cluster ai2/saturn-cirrascale --gpus 8 --name SiTXL2-StackedCFG-Ours-cfg-1.6-0.65-50 -- bash eval_script_with_cfg_stacked_ours.sh 1.6 0.65 50
gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --cluster ai2/saturn-cirrascale --gpus 8 --name SiTXL2-StackedCFG-Ours-cfg-1.4-0.65-50 -- bash eval_script_with_cfg_stacked_ours.sh 1.4 0.65 50
gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --cluster ai2/saturn-cirrascale --gpus 8 --name SiTXL2-StackedCFG-Ours-cfg-1.2-0.65-50 -- bash eval_script_with_cfg_stacked_ours.sh 1.2 0.65 50
gantry run $args --workspace ai2/structured_diffusion --cluster ai2/jupiter-cirrascale-2 --cluster ai2/saturn-cirrascale --gpus 8 --name SiTXL2-StackedCFG-Ours-cfg-1.0-0.65-50 -- bash eval_script_with_cfg_stacked_ours.sh 1.0 0.65 50
