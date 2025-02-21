#!/bin/bash
MODEL_NAME="SiT-XL/2"
MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05 linear-dinov2-b-enc8"
CHECKPOINT_FNAME="0400000.pt"
STEPS="5 10 20 30 100 150 200"


for steps in $STEPS
do
    for exp_name in $MODEL_ITERS
        do
        EXP_LOC="/weka/prior-default/georges/research/REPA/exps/${exp_name}"
        SAVE_DIR="/weka/prior-default/georges/research/REPA/samples/fid_50k/${steps}_steps/${exp_name}"
        if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then 
            torchrun \
            --nnodes=1 \
            --nproc_per_node=3 \
            --master-port 29501 \
            generate.py \
            --model "${MODEL_NAME}" \
            --num-fid-samples 50000 \
            --ckpt "${EXP_LOC}/checkpoints/${CHECKPOINT_FNAME}" \
            --path-type=linear \
            --encoder-depth=8 \
            --projector-embed-dims=768 \
            --per-proc-batch-size=64 \
            --mode=sde \
            --num-steps=$steps \
            --cfg-scale=1.0 \
            --guidance-high=0.0 \
            --sample-dir "${SAVE_DIR}"
        fi
    done 
done
