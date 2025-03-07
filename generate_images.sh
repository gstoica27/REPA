#!/bin/bash
MODEL_NAME="SiT-XL/2"
# MODEL_ITERS="repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripanyTemp0p025-res512"
MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-meanTemp0p0-res512"

CHECKPOINT_FNAME="0400000.pt"
STEPS="50"


for steps in $STEPS
do
    for exp_name in $MODEL_ITERS
        do
        EXP_LOC="/weka/oe_training_default/georges/checkpoints/REPA/exps/exps2/${exp_name}"
        SAVE_DIR="/weka/oe_training_default/georges/research/REPA/samples_res512_teaser/${steps}_steps/${exp_name}"
        if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then 
            torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
            --master-port 29501 \
            generate_with_intermediates.py \
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
            --sample-dir "${SAVE_DIR}" \
            --record-custom-classes 337 250 972 90 29 297 248 973 993 385 387 417 279 291 599 985 107 125 812 809 616 644 643 408 968 475 549 975 323 280 9 277 586 360 387 974 88 979 279\
            --rough-examples-per-class 128 \
            --resolution 512
            # --record-intermediate-steps \
            # --record-intermediate-steps-freq 5 \
        fi
    done 
done
