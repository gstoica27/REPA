import pdb
import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

with open('/weka/prior-default/georges/keys/wandb.txt', 'r') as f:
    wandb_key = f.readlines()[0].strip()
wandb.login(key=wandb_key) # login to wandb
if not os.path.exists('/root/.cache/torch/hub/facebookresearch_dinov2_main'):
    from distutils.dir_util import copy_tree
    copy_tree(
        '/weka/prior-default/georges/redundancies/facebookresearch_dinov2_main', 
        '/root/.cache/torch/hub/facebookresearch_dinov2_main'
    )
# os.makedirs('/root/.cache/torch/hub/facebookresearch_dinov2_main', exist_ok=True)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0., return_mean=True):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    if return_mean:
        z = mean
    else:
        z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    # logging_dir = Path(args.output_dir, args.logging_dir)
    # accelerator_project_config = ProjectConfiguration(
    #     project_dir=args.output_dir, logging_dir=logging_dir
    #     )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # project_config=accelerator_project_config,
    )

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    # if args.enc_type != None:
    #     encoders, encoder_types, architectures = load_encoders(
    #         args.enc_type, device, args.resolution
    #         )
    # else:
    #     raise NotImplementedError()
    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = [768], # NOTE: HACK to have something to take latent dim out of 
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
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    loss_fn = torch.nn.CrossEntropyLoss()
    if accelerator.is_main_process:
        # logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # overwrite projectors so I can use them in Linear Probing
    model.projectors = nn.ModuleList([nn.Identity() for z_dim in model.z_dims])
    model = model.to(device)
    model.eval()

    probe = torch.nn.Linear(model.hidden_size, args.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
    )

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # Setup data:
    train_dataset = CustomDataset(args.train_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    eval_dataset = CustomDataset(args.eval_dir)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if accelerator.is_main_process:
        # logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
        print(f"Train Dataset contains {len(train_dataset):,} Train Images ({args.train_dir})")
        print(f"Eval Dataset contains {len(eval_dataset):,} Train Images ({args.eval_dir})")
    
    
    model, optimizer, train_dataloader, eval_dataloader, probe = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, probe
    )

    # if accelerator.is_main_process:
    #     tracker_config = vars(copy.deepcopy(args))
    #     accelerator.init_trackers(
    #         project_name=args.wandb_project, 
    #         config=tracker_config,
    #         init_kwargs={
    #             "wandb": {"name": f"{exp_name}"}
    #         },
    #     )
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            # pdb.set_trace()
            # raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias, return_mean=True)
            y = y.to(device)
            z = None
            labels = y

            with accelerator.accumulate(probe):
                _, latents_list, altered_labels = model(
                    x, torch.zeros(x.shape[0]).to(device), y=labels
                )
                # pdb.set_trace()
                latents = latents_list[0].mean(dim=1) # [N, D]
                preds = probe(latents)
                loss = loss_fn(preds, labels)
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = probe.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    print(
                        f"Train Epoch: {epoch} {progress_message}\t"
                        f"Loss: {loss.item():.6f}"
                        f"LR {optimizer.param_groups[0]['lr']:.5f}"
                    )

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    probe.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    total_correct = 0
    total_samples = 0
    print("Evaluating Probe on Eval Dataset...")
    for _, x, y in eval_dataloader:
        with torch.no_grad():
            x = x.squeeze(dim=1).to(device)
            _, latents_list = model(
                x, torch.zeros(x.shape[0]), y=labels
            )
            latents = latents_lists[0].mean(dim=1) # [N, D]
            preds = probe(latents).argmax(dim=-1)
            batch_correct = preds == y.to(device)

            total_correct += batch_correct.sum()
            total_samples += x.shape[0]
    
    print(f"Final Evaluation on Probe at Encoding Layer {model.encoder_depth}: {total_correct / total_samples * 100:.3f}%")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # logger.info("Done!")
        print("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    # parser.add_argument("--output-dir", type=str, default="exps")
    # parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    # parser.add_argument("--sampling-steps", type=int, default=10000)
    # parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="REPA_PROBE")

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")

    # dataset
    parser.add_argument("--train-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--eval-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    # parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    # parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    # parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    # parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    # parser.add_argument("--cfg-prob", type=float, default=0.1)
    # parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    # parser.add_argument("--proj-coeff", type=float, default=0.5)
    # parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    
    # # Structure Loss
    # parser.add_argument("--struct-coeff", type=float, default=0.0)
    # parser.add_argument('--struct-method', type=str, default=None, choices=[None, "between_images", "between_tokens", "between_images_per_token"])
    # parser.add_argument('--struct-add-relu', action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument('--struct-encoder-depth', type=int, default=8)
    
    # Additional terms
    # parser.add_argument(
    #     "--denoising-type", 
    #     type=str, 
    #     default="mean", 
    #     # choices=[
    #     #     "mean", 
    #     #     "triplet_any_noise", 
    #     #     "class_conditioned_triplet_mse",
    #     #     "triplet_same_noise"
    #     # ]
    # )
    # parser.add_argument("--denoising-temp", default=1.0, type=float, help="Temperature for the denoising loss.")
    # parser.add_argument("--dont-contrast-on-unconditional", action=argparse.BooleanOptionalAction, default=False,
    #                     help="If True, apply contrastive loss on unconditional samples.")
    # parser.add_argument("--is-class-conditioned", action=argparse.BooleanOptionalAction, default=False, 
    #                     help="If True, apply class conditioning for triplet loss (only for triplet loss). ")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    print("The args are: ", args)

    import torch.distributed as dist
    
    if dist.is_initialized():
        if dist.get_rank() == 0:
            while True:
                try:
                    torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
                except Exception as e:
                    print(e)
                    continue
                break
        dist.barrier()
    
    # import shutil
    # if not os.path.exists('/root/.cache/torch/hub/facebookresearch_dinov2_main/hubconf.py'):
    #     print("Creating symlink...")
    #     shutil.copytree(
    #         '/weka/prior-default/georges/redundancies/facebookresearch_dinov2_main',
    #         '/root/.cache/torch/hub/facebookresearch_dinov2_main',
    #         dirs_exist_ok=True,
    #         symlinks=True,
    #     )
    # exp_name = create_experiment_name(args)
    # print("The experiment name is: ", exp_name)
    # try:
    #     main(args, exp_name)
    # except:
    #     print("Retrying....")
    #     main(args, exp_name)
    main(args)









