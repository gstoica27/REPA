#!/bin/bash
# MODEL_NAME="SiT-B/2"
MODEL_NAME="SiT-XL/2"
# MODEL_ITERS="repaLinear-0p5-sitb2-dinov2VitB-enc4-bs512 repaLinear-0p5-sitb2-dinov2VitB-enc4-bs512-tripmseTemp0p05 repaLinear-0p5-sitb2-dinov2VitB-enc4-bs1024 repaLinear-0p5-sitb2-dinov2VitB-enc4-bs1024-tripanyTemp0p1"
MODEL_ITERS="repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05"
CFG_BEGIN=4.25
CFG_END=4.0
CFG_STEP=8.25

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
    # if [ ! -d "${SAVE_DIR}" ]; then
    for cfg_scale in $(seq $CFG_BEGIN $CFG_STEP $CFG_END); do
        SAVE_DIR="/weka/prior-default/georges/research/REPA/samples/fid_100/250_steps/with_cfg/${exp_name}"
        torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        --master-port 29502 \
        generate.py \
        --model "${MODEL_NAME}" \
        --num-fid-samples 100 \
        --ckpt "${EXP_LOC}/checkpoints/0400000.pt" \
        --path-type=linear \
        --encoder-depth=8 \
        --projector-embed-dims=768 \
        --per-proc-batch-size=64 \
        --mode=sde \
        --num-steps=250 \
        --cfg-scale=$cfg_scale \
        --guidance-high=0.7 \
        --sample-dir "${SAVE_DIR}"
    done 
done