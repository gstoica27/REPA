#!/bin/bash
MODEL_NAME="SiT-XL/2"
MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05 linear-dinov2-b-enc8"
CHECKPOINT_FNAME="0400000.pt"
STEPS="50"
TRAJECTORY_STRUCTURE_TYPES="segment_cosine source_cosine"


for steps in $STEPS
do
    for exp_name in $MODEL_ITERS
        do
        EXP_LOC="/weka/prior-default/georges/research/REPA/exps2/${exp_name}"
        SAVE_DIR="/weka/prior-default/georges/research/REPA/samples_analysis/${steps}_steps/${exp_name}"
        for trajectory_structure_type in $TRAJECTORY_STRUCTURE_TYPES 
        do
            if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then 
                torchrun \
                --nnodes=1 \
                --nproc_per_node=8 \
                --master-port 29501 \
                generate_with_intermediates.py \
                --model "${MODEL_NAME}" \
                --num-fid-samples 10000 \
                --ckpt "${EXP_LOC}/checkpoints/${CHECKPOINT_FNAME}" \
                --path-type=linear \
                --encoder-depth=8 \
                --projector-embed-dims=768 \
                --per-proc-batch-size=64 \
                --mode=sde \
                --num-steps=$steps \
                --cfg-scale=1.0 \
                --guidance-high=0.0 \
                --sample-dir "${SAVE_DIR}" \
                --record-trajectory-structure \
                --trajectory-structure-type $trajectory_structure_type
            fi
        done
    done 
done
