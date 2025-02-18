#!/bin/bash
MODEL_NAME="SiT-B/2"
# Compute this list using the utils find_experiment_paths -> convert_pylist_to_shlist functions! 
# for exp_name in "structImg-0p5-repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256" "structImg-noRelu-0p5-repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256" "structTok-0p5-repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256" "structTok-noRelu-0p5-repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256" "structImgByTok-0p5-repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256" "structImgByTok-noRelu-0p5-repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256" 
# for exp_name in "repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripmseTemp0p1" "repaLinear-0p5-sitb2-dinov2VitB-enc4-bs512-tripmseTemp0p05" "linear-sitb-dinov2-b-enc4/between_images-structCoeff_0.0" "repaLinear-0p5-sitb2-dinov2VitB-enc4-bs512"
for exp_name in "repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripsameTemp0p1" "repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripanyTemp0p1" "repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripsameTemp0p1" "linear-dinov2-b-enc8" "repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripsameTemp0p05" "repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05"
do
    # EXP_LOC="exps/${exp_name}"
    EXP_LOC="/weka/prior-default/georges/research/REPA/exps/${exp_name}"
    SAVE_DIR="/weka/prior-default/georges/research/REPA/samples/fid_50k/50_steps/${exp_name}"
    if [ ! -d "${SAVE_DIR}" ]; then
        for fname in "0050000.pt" "0100000.pt" "0150000.pt" "0200000.pt" "0250000.pt" "0300000.pt" "0350000.pt" "0400000.pt"
        # for fname in "0400000.pt"
            do 
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
                --sample-dir "${SAVE_DIR}"
            done 
    fi
done

for exp_name in "repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripsameTemp0p1" "repaLinear-0p5-sitb2-dinov2VitB-enc4-bs256-tripanyTemp0p1" "repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripsameTemp0p1" "repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripsameTemp0p05" "repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05"
do
    # EXP_LOC="exps/${exp_name}"
    EXP_LOC="/weka/prior-default/georges/research/REPA/exps/${exp_name}"
    SAVE_DIR="/weka/prior-default/georges/research/REPA/samples/fid_50k/250_steps/${exp_name}"
    if [ ! -d "${SAVE_DIR}" ]; then
        for fname in "0050000.pt" "0100000.pt" "0150000.pt" "0200000.pt" "0250000.pt" "0300000.pt" "0350000.pt" "0400000.pt"
        # for fname in "0400000.pt"
            do 
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
                --num-steps=250 \
                --cfg-scale=1.0 \
                --guidance-high=0.0 \
                --sample-dir "${SAVE_DIR}"
            done 
    fi
done