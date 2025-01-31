for fname in "0050000.pt" "0100000.pt" "0150000.pt" "0200000.pt" "0250000.pt" "0300000.pt" "0350000.pt" "0400000.pt"
    do 
        torchrun --nnodes=1 --nproc_per_node=8 generate.py --model SiT-XL/2 --num-fid-samples 10000 --ckpt "exps/linear-dinov2-b-enc8/checkpoints/${fname}" --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=64 --mode=sde --num-steps=50 --cfg-scale=1.0 --guidance-high=0.0 --sample-dir samples/linear-dinov2-b-enc8
    done