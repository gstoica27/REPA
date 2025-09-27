#!/bin/bash
MODEL_NAME="SiT-B/2"
# MODEL_NAME="SiT-XL/2"
MODEL_ITERS="repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-cbcTemp0p25-res256"
CHECKPOINT_ITERS="0400000.pt"
SAVE_DIR="/weka/oe-training-default/georges/samples/contrast_by_class/fid_50k"
# EXP_LOC="/weka/oe-training-default/georges/checkpoints/REPA/exps/"
EXP_LOC="/weka/oe-training-default/georges/checkpoints/REPA/exps/contrast_by_class2"

for exp_name in $MODEL_ITERS
do 
    for fname in $CHECKPOINT_ITERS
    do
        if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then   
            echo $exp_name $fname
        fi
    done 
done

for exp_name in $MODEL_ITERS
do
    LOAD_EXP_LOC="${EXP_LOC}/${exp_name}"
    EXP_SAVE_DIR="${SAVE_DIR}/50_steps/${exp_name}"
    for fname in $CHECKPOINT_ITERS
        do
            torchrun \
            --nnodes=1 \
            --nproc_per_node=4 \
            --master-port 29501 \
            generate.py \
            --model "${MODEL_NAME}" \
            --num-fid-samples 50000 \
            --ckpt "${LOAD_EXP_LOC}/checkpoints/${fname}" \
            --path-type=linear \
            --encoder-depth=8 \
            --projector-embed-dims=768 \
            --per-proc-batch-size=64 \
            --mode=sde \
            --num-steps=50 \
            --cfg-scale=1.0 \
            --guidance-high=0.0 \
            --sample-dir "${EXP_SAVE_DIR}"
        # fi
    done 
done