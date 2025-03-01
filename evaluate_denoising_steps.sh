#!/bin/bash
MODEL_NAME="SiT-XL/2"
# MODEL_ITERS="linear-dinov2-b-enc8"
MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05"
CHECKPOINT_FNAME="0400000.pt"
STEPS="50"


for steps in $STEPS
do
    for exp_name in $MODEL_ITERS
        do
        EXP_LOC="/weka/prior-default/georges/research/REPA/exps2/${exp_name}"
        SAVE_DIR="/weka/prior-default/georges/research/REPA/samples_analysis/fid_50k/varying_steps/${steps}_steps/${exp_name}"
        if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then 
            torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
            --master-port 29501 \
            generate_with_intermediates.py \
            --model "${MODEL_NAME}" \
            --num-fid-samples 1000 \
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
            --record-intermediate-steps \
            --record-intermediate-steps-freq 10 \
            --record-custom-classes 153 88 2 417 933 555 932 385 386 294 33 207 250 387 402 620 812 429 388 360 291 89 928 973 302 340 511 609 878 919 888 698\
            --rough-examples-per-class 64
        fi
    done 
done
