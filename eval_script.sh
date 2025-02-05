#!/bin/bash
MODEL_NAME="SiT-B/2"
# Compute this list using the utils find_experiment_paths -> convert_pylist_to_shlist functions! 
for exp_name in "structImg-noRelu-0p5-repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256"
do
    EXP_LOC="exps/${exp_name}"
    SAVE_DIR="samples/${exp_name}"
    if [ ! -d "${SAVE_DIR}" ]; then
        for fname in "0050000.pt" "0100000.pt" "0150000.pt" "0200000.pt" "0250000.pt" "0300000.pt" "0350000.pt" "0400000.pt"
            do 
                torchrun --nnodes=1 --nproc_per_node=8 generate.py --model "${MODEL_NAME}" --num-fid-samples 10000 --ckpt "${EXP_LOC}/checkpoints/${fname}" --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=64 --mode=sde --num-steps=50 --cfg-scale=1.0 --guidance-high=0.0 --sample-dir "${SAVE_DIR}"
            done 
    fi
done