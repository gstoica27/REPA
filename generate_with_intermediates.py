# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model
import pdb
import random
from imnet1k_classes import IMNET_CLS_DICT
import torch.nn.functional as F

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def create_npz_from_latents(sample_dir, latent_idxs, latent_samples, num=50_000):
    """
    Builds a single .npz file from a list of latents.
    """
    latents = []
    for i in tqdm(range(num), desc="Building .npz file from latents"):
        latents.append(latent_samples[latent_idxs.index(i)])
    latents = np.stack(latents)
    assert latents.shape == (num, latents.shape[1])
    npz_path = f"{sample_dir}_latents.npz"
    np.savez(npz_path, arr_0=latents)
    print(f"Saved .npz file to {npz_path} [shape={latents.shape}].")
    return npz_path

def create_image_from_latents(vae, samples, latents_bias, latents_scale):
    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
    samples = (samples + 1) / 2.
    samples = torch.clamp(
        255. * samples, 0, 255
    ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return samples

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        # args.save_latents:
        #     latents_folder_dir = f"{args.sample_dir}/{folder_name}/latents"
        # os.makedirs(latents_folder_dir, exist_ok=True)
        # print(f"Saving latents at {latents_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    
    if args.record_custom_classes is not None and len(args.record_custom_classes) > 0:
        rough_examples_per_class = args.rough_examples_per_class if args.rough_examples_per_class is not None else args.per_proc_batch_size
        assert rough_examples_per_class % dist.get_world_size() == 0, "rough_examples_per_class must be divisible by world_size"
        total_samples = int(len(args.record_custom_classes) * rough_examples_per_class)
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        # Assign the actual batch size to be used for each GPU
        if samples_needed_this_gpu % args.per_proc_batch_size == 0:
            n = args.per_proc_batch_size
        elif samples_needed_this_gpu % args.rough_examples_per_class == 0:
            n = args.rough_examples_per_class
        elif samples_needed_this_gpu <= args.per_proc_batch_size:
            n = samples_needed_this_gpu
        else:
            raise ValueError("samples_needed_this_gpu must be divisible by the per-GPU batch size or the rough_examples_per_class")
        global_batch_size = n * dist.get_world_size()
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")    
            print("Samples needed this GPU: ", samples_needed_this_gpu)
            print("N: ", n)
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
        iterations = int(samples_needed_this_gpu // n)
    else:
        n = args.per_proc_batch_size
        global_batch_size = n * dist.get_world_size()
        # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
        total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")
            print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
        assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
        iterations = int(samples_needed_this_gpu // n)
    
    if args.record_trajectory_structure:
        trajectory_idxs = {}
        
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        
        if args.record_custom_classes is not None and len(args.record_custom_classes) > 0:
            y = torch.tensor(np.random.choice(args.record_custom_classes, n, replace=True), device=device)
        else:
            y = torch.randint(0, args.num_classes, (n,), device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            record_intermediate_steps=args.record_intermediate_steps,
            record_intermediate_steps_freq=args.record_intermediate_steps_freq,
            record_trajectory_structure=args.record_trajectory_structure,
            trajectory_structure_type=args.trajectory_structure_type,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples_dict = euler_maruyama_sampler(**sampling_kwargs)
            elif args.mode == "ode":
                    # samples, intermediate_steps = euler_sampler(**sampling_kwargs)
                samples_dict = euler_sampler(**sampling_kwargs)
            else:
                raise NotImplementedError()
            latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
            
            samples = samples_dict['samples'].to(torch.float32)
            if args.record_intermediate_steps:
                intermediate_steps = samples_dict['intermediate_steps']
                intermediate_images = [create_image_from_latents(vae, intermediate_step, latents_bias, latents_scale) for intermediate_step in intermediate_steps]
                intermediate_samples = np.stack(intermediate_images, axis=0).transpose((1,0,2,3,4))
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            # Save samples to disk as individual .png files
            # for i, sample in enumerate(samples):
            #     index = i * dist.get_world_size() + rank + total
            #     Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
                
            if args.record_intermediate_steps:
                # Save images from intermediate steps
                for i, (final_sample, path_images) in enumerate(zip(samples, intermediate_samples)):
                    index = i * dist.get_world_size() + rank + total
                    cls_id = y[i].item()
                    cls_name = IMNET_CLS_DICT[cls_id]
                    save_dir = os.path.join(sample_folder_dir, "intermediate_steps", cls_name)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(final_sample).save(f"{save_dir}/{index:06d}.png")
                    
                    intermediates_save_dir = os.path.join(save_dir, f"{index:06d}_path")
                    os.makedirs(intermediates_save_dir, exist_ok=True)
                    for delta, image_in_path in enumerate(path_images):
                        interval = (delta+1) * args.record_intermediate_steps_freq
                        save_path = os.path.join(intermediates_save_dir, f"step_{interval}.png")
                        Image.fromarray(image_in_path).save(save_path)

            if args.record_trajectory_structure:
                trajectory_vectors = F.normalize(samples_dict['trajectory_vectors'].transpose(1, 0).flatten(2), dim=-1)
                if args.trajectory_structure_type == "segment_cosine":
                    A = trajectory_vectors[:,:-1,:]
                    B = trajectory_vectors[:, 1:,:]
                    similarities = (A * B).sum(dim=-1)
                elif args.trajectory_structure_type == "source_cosine":
                    similarities = torch.bmm(trajectory_vectors, trajectory_vectors.transpose(1, 2))
                    similarities.diagonal(dim1=1, dim2=2).fill_(0) # zero-out the identical samples
                else:
                    raise ValueError("Invalid trajectory_structure_type")
                
                similarities = similarities.cpu()
                for i, sample_sims in enumerate(similarities):
                    index = i * dist.get_world_size() + rank + total
                    trajectory_idxs[index] = sample_sims
        
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        print("Length of trajectory: ", len(trajectory_idxs))
        print("Trajectory keys: ", list(trajectory_idxs.keys()))
        print("Number of samples: ", arg.num_fid_samples)
        selected_samples = torch.stack([trajectory_idxs[i] for i in range(args.num_fid_samples)]).numpy()
        np.savez(f"{sample_folder_dir}_trajectory_{args.trajectory_structure_type}.npz", arr_0=selected_samples)
    #     create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode
    
    # Instructions for recording intermediate path trajectories
    parser.add_argument("--record-intermediate-steps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-intermediate-steps-freq", type=int, default=10)
    parser.add_argument("--record-custom-classes", type=int, default=None, nargs='+')
    parser.add_argument("--rough-examples-per-class", type=int, default=32)
    
    # Instructions for computing cosine similarities
    parser.add_argument("--record-trajectory-structure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trajectory-structure-type", type=str, default=None, choices=["segment_cosine", "source_cosine", None])

    args = parser.parse_args()
    main(args)
