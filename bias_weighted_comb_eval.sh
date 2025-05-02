# !/bin/bash
bias_lambda=$1
velocity_lambda=$2

 gantry run \
    --budget ai2/prior \
    --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-CFG-NFE-200-BiasLambda$bias_lambda-VelLambda$velocity_lambda" \
    --priority high \
    --gpus 8 \
    --weka "prior-default:/weka/prior-default" \
    --weka "oe-training-default:/weka/oe-training-default" \
    --conda environment.yml \
    --workspace ai2/structured_diffusion \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --retries 3 \
    -- torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master-port 29501 \
    generate.py \
    --model SiT-XL/2 \
    --num-fid-samples 50000 \
    --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
    --path-type=linear \
    --encoder-depth=8 \
    --projector-embed-dims=768 \
    --per-proc-batch-size=64 \
    --mode=sde \
    --num-steps=200 \
    --cfg-scale=1.85 \
    --guidance-high=0.65 \
    --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2 \
    --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
    --bias-weight $bias_lambda \
    --velocity-weight $velocity_lambda \
    --debias-method weighted \
    --subtract-bias

gantry run \
    --budget ai2/prior \
    --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-CFG-NFE-200-BiasLambda$bias_lambda-VelLambda$velocity_lambda" \
    --priority high \
    --gpus 8 \
    --weka "prior-default:/weka/prior-default" \
    --weka "oe-training-default:/weka/oe-training-default" \
    --conda environment.yml \
    --workspace ai2/structured_diffusion \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --retries 3 \
    -- torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master-port 29501 \
    generate.py \
    --model SiT-XL/2 \
    --num-fid-samples 50000 \
    --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
    --path-type=linear \
    --encoder-depth=8 \
    --projector-embed-dims=768 \
    --per-proc-batch-size=64 \
    --mode=sde \
    --num-steps=200 \
    --cfg-scale=1.85 \
    --guidance-high=0.65 \
    --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2 \
    --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
    --bias-weight $bias_lambda \
    --velocity-weight $velocity_lambda \
    --debias-method weighted