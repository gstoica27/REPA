#!/bin/bash
MODEL_NAME="SiT-B/2"
# MODEL_NAME="SiT-XL/2"
# MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p001 repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p01 repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p15"
MODEL_ITERS="repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-meanTemp0p0-res512 repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripanyTemp0p05-res512 repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripsameTemp0p05-res512"
# CHECKPOINT_ITERS="0050000.pt 0100000.pt 0150000.pt 0200000.pt 0250000.pt 0300000.pt 0350000.pt 0400000.pt"
CHECKPOINT_ITERS="0400000.pt"
# Compute this list using the utils find_experiment_paths -> convert_pylist_to_shlist functions! 
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
    # EXP_LOC="exps/${exp_name}"
    EXP_LOC="/weka/prior-default/georges/research/REPA/exps2/${exp_name}"
    SAVE_DIR="/weka/prior-default/georges/research/REPA/samples/fid_50k/50_steps/${exp_name}"
        # for fname in "0050000.pt" "0100000.pt" "0150000.pt" "0200000.pt" "0250000.pt" "0300000.pt" "0350000.pt" "0400000.pt"
    for fname in $CHECKPOINT_ITERS
        do
        if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then 
            torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
            --master-port 29501 \
            generate.py \
            --model "${MODEL_NAME}" \
            --num-fid-samples 50000 \
            --ckpt "${EXP_LOC}/checkpoints/${fname}" \
            --path-type=linear \
            --encoder-depth=8 \
            --projector-embed-dims=768 \
            --per-proc-batch-size=64 \
            --mode=sde \
            --num-steps=50 \
            --cfg-scale=1.0 \
            --guidance-high=0.0 \
            --sample-dir "${SAVE_DIR}" \
            --resolution 512
        fi
    done 
done

for exp_name in $MODEL_ITERS
do
    # EXP_LOC="exps/${exp_name}"
    EXP_LOC="/weka/prior-default/georges/research/REPA/exps2/${exp_name}"
    SAVE_DIR="/weka/prior-default/georges/research/REPA/samples/fid_50k/250_steps/${exp_name}"
    # if [ ! -d "${SAVE_DIR}" ]; then
    for fname in $CHECKPOINT_ITERS
    # for fname in "0400000.pt"
        do
        if [ ! -d "${SAVE_DIR}/${exp_name}" ]; then  
            torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
            --master-port 29502 \
            generate.py \
            --model "${MODEL_NAME}" \
            --num-fid-samples 50000 \
            --ckpt "${EXP_LOC}/checkpoints/${fname}" \
            --path-type=linear \
            --encoder-depth=8 \
            --projector-embed-dims=768 \
            --per-proc-batch-size=64 \
            --mode=sde \
            --num-steps=250 \
            --cfg-scale=1.0 \
            --guidance-high=0.0 \
            --sample-dir "${SAVE_DIR}" \
            --resolution 512
        fi
    done 
done