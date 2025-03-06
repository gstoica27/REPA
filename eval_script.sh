#!/bin/bash
# MODEL_NAME="SiT-B/2"
MODEL_NAME="SiT-XL/2"
# MODEL_ITERS="linear-sitb-dinov2-b-enc4/between_images-structCoeff_0.0 repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripmseTemp0p5"
# MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05-res256/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-meanTemp0p0-res256"
MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05-res256"
# CHECKPOINT_ITERS="0050000.pt 0100000.pt 0150000.pt 0200000.pt 0250000.pt 0300000.pt 0350000.pt 0400000.pt"
CHECKPOINT_ITERS="0050000.pt 0100000.pt 0150000.pt 0200000.pt 0250000.pt 0300000.pt 0350000.pt 0400000.pt 0450000.pt 0500000.pt 0550000.pt 0600000.pt 0650000.pt 0700000.pt 0750000.pt 0800000.pt 0850000.pt 0900000.pt 0950000.pt 1000000.pt 1050000.pt 1100000.pt 1150000.pt 1200000.pt 1250000.pt 1300000.pt 1350000.pt 1400000.pt 1450000.pt 1500000.pt 1550000.pt 1600000.pt 1650000.pt 1700000.pt 1750000.pt 1800000.pt"
# CHECKPOINT_ITERS="0400000.pt"
# Compute this list using the utils find_experiment_paths -> convert_pylist_to_shlist functions! 
# SAVE_DIR="/weka/prior-default/georges/research/REPA/samples/fid_50k"
SAVE_DIR="/weka/oe_training_default/georges/research/REPA/samples2/fid_50k/7M_models"
EXP_LOC="/weka/oe_training_default/georges/checkpoints/REPA/exps/exps2_7M"

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
        # if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then 
            torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
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

for exp_name in $MODEL_ITERS
do
    # EXP_LOC="exps/${exp_name}"
    LOAD_EXP_LOC="${EXP_LOC}/${exp_name}"
    EXP_SAVE_DIR="${SAVE_DIR}/250_steps/${exp_name}"
    for fname in $CHECKPOINT_ITERS
        do
        if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then  
            torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
            --master-port 29502 \
            generate.py \
            --model "${MODEL_NAME}" \
            --num-fid-samples 50000 \
            --ckpt "${LOAD_EXP_LOC}/checkpoints/${fname}" \
            --path-type=linear \
            --encoder-depth=8 \
            --projector-embed-dims=768 \
            --per-proc-batch-size=64 \
            --mode=sde \
            --num-steps=250 \
            --cfg-scale=1.0 \
            --guidance-high=0.0 \
            --sample-dir "${EXP_SAVE_DIR}"
        fi
    done 
done