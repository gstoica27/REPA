#!/bin/bash
# MODEL_NAME="SiT-B/2"
MODEL_NAME="SiT-XL/2"
# MODEL_ITERS="sitxl2-vanilla-bs256-tripanyTemp0p05"
# MODEL_ITERS="sitxl2-vanilla-bs256-meanTemp0p0-res256"
MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripmseTemp0p05"
# MODEL_ITERS="linear-dinov2-b-enc8"
# MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-cctripanyTemp0p05-res256"
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
    EXP_LOC="/weka/oe-training-default/georges/checkpoints/REPA/exps/${exp_name}"
    # EXP_LOC="/weka/oe-training-default/georges/checkpoints/REPA/exps/exps2/${exp_name}"
    # SAVE_DIR="/weka/oe-training-default/georges/samples/vanilla/fid_50k/"$3"_steps/with_cfg/with_cfg_"$1"_"$2"/${exp_name}"
    SAVE_DIR="/weka/oe-training-default/georges/samples/ode/repa/fid_50k/"$3"_steps/with_cfg/with_cfg_"$1"_"$2"/${exp_name}"
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
            --mode=ode \
            --num-steps=$3 \
            --cfg-scale=$1 \
            --guidance-high=$2 \
            --heun \
            --sample-dir "${SAVE_DIR}"
        fi
    done 
done

