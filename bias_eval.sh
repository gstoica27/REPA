# !/bin/bash
# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-BL-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-BL-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.05

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-Ours-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.8 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p8-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.05

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-Ours-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.8 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p8-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-Ours-CFG-NFE-200" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-Ours-CFG-NFE-200" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.05

# Raw Bias
# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-BL-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-raw-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_raw_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-BL-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-raw-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_raw_bias.pt \
#     --bias-weight 0.05

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-Ours-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.8 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-raw-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p8-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_raw_bias.pt \
#     --bias-weight 0.05

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-Ours-CFG" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.8 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-raw-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p8-guidance-0p65 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_raw_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-Ours-CFG-NFE-200" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-raw-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_raw_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-Ours-CFG-NFE-200" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-raw-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_raw_bias.pt \
#     --bias-weight 0.05

# Sampled Latent Bias
# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Sampled-Latent-Bias-BL-CFG-NFE50" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-sampled-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-50 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_sampled_latent_bias.pt \
#     --bias-weight 0.05


# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Sampled-Latent-Bias-BL-CFG-NFE50" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-sampled-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-50 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_sampled_latent_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias


# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Sampled-Latent-Bias-Ours-CFG-NFE50" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.8 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-sampled-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p8-guidance-0p65-nfe-50 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_sampled_latent_bias.pt \
#     --bias-weight 0.05 


# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Sampled-Latent-Bias-Ours-CFG-NFE50" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=50 \
#     --cfg-scale=1.8 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-sampled-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p8-guidance-0p65-nfe-50 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_sampled_latent_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias


# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Sampled-Latent-Bias-BL-CFG-NFE200" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-sampled-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_sampled_latent_bias.pt \
#     --bias-weight 0.05


# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Subtract-Sampled-Latent-Bias-BL-CFG-NFE200" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/subtract-sampled-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_sampled_latent_bias.pt \
#     --bias-weight 0.05 \
#     --subtract-bias

# Additional Searches Around Best Result
# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-Ours-CFG-NFE-200-Lambda0p1" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --cluster ai2/jupiter-cirrascale-2 \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200-lambda0p1 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.1

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-Ours-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-Ours-CFG-NFE-200-Lambda0p15" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --cluster ai2/jupiter-cirrascale-2 \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200-lambda0p15 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.15

# gantry run \
#     --budget ai2/prior \
#     --name "SiTXL2-BL-Is-Bias-The-Answer-FID50K-Eval-Add-Latent-Bias-Ours-CFG-NFE-200-Lambda0p05" \
#     --priority high \
#     --gpus 8 \
#     --weka "prior-default:/weka/prior-default" \
#     --weka "oe-training-default:/weka/oe-training-default" \
#     --conda environment.yml \
#     --workspace ai2/structured_diffusion \
#     --cluster ai2/saturn-cirrascale \
#     --cluster ai2/jupiter-cirrascale-2 \
#     --retries 3 \
#     --not-preemptible \
#     -- torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 29501 \
#     generate.py \
#     --model SiT-XL/2 \
#     --num-fid-samples 50000 \
#     --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/linear-dinov2-b-enc8/checkpoints/0400000.pt \
#     --path-type=linear \
#     --encoder-depth=8 \
#     --projector-embed-dims=768 \
#     --per-proc-batch-size=64 \
#     --mode=sde \
#     --num-steps=200 \
#     --cfg-scale=1.85 \
#     --guidance-high=0.65 \
#     --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200-lambda0p15 \
#     --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
#     --bias-weight 0.05 \
#     --is-baseline

gantry run \
    --budget ai2/prior \
    --name "SiTXL2-BL-Is-Bias-The-Answer-FID50K-Eval-Subtract-Latent-Bias-Ours-CFG-NFE-200-Lambda0p05" \
    --priority low \
    --gpus 8 \
    --weka "prior-default:/weka/prior-default" \
    --weka "oe-training-default:/weka/oe-training-default" \
    --conda environment.yml \
    --workspace ai2/structured_diffusion \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/rhea-cirrascale \
    --retries 3 \
    -- torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master-port 29501 \
    generate.py \
    --model SiT-XL/2 \
    --num-fid-samples 50000 \
    --ckpt /weka/oe-training-default/georges/checkpoints/REPA/exps/linear-dinov2-b-enc8/checkpoints/0400000.pt \
    --path-type=linear \
    --encoder-depth=8 \
    --projector-embed-dims=768 \
    --per-proc-batch-size=64 \
    --mode=sde \
    --num-steps=200 \
    --cfg-scale=1.85 \
    --guidance-high=0.65 \
    --sample-dir /weka/oe-training-default/georges/samples/biased-cfg2/add-latent-bias/repaLinear-0p5-sitxl2-dinov2VitB-enc8-bs256-tripanyTemp0p05/cfg-1p85-guidance-0p65-nfe-200-lambda0p15 \
    --bias-path /weka/prior-default/georges/research/REPA/biases/imnet256_latent_bias.pt \
    --bias-weight 0.05 \
    --is-baseline \
    --subtract-bias
    