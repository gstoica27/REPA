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
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# from accelerate import Accelerator
import torch.distributed as dist
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

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

class ComputeVelocityCosSims:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            null_class_idx=None,
            ):
        self.weighting = weighting
        self.path_type = path_type
        self.null_class_idx = null_class_idx
        assert self.null_class_idx is not None, "Null class index must be provided"
    
    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_negatives(self, y, labels=None):
        bsz = y.shape[0]
        choices = torch.tile(torch.arange(bsz), (bsz, 1)).to(y.device)
        choices.fill_diagonal_(-1.)
        choices = choices.sort(dim=1)[0][:, 1:]
        choices = choices[torch.arange(bsz), torch.randint(0, bsz-1, (bsz,))]
        y_neg = y[choices]
        
        non_nulls = labels != self.null_class_idx
        
        return y_neg, non_nulls
    
    def get_pairs(self, target_images, d_alpha_t, d_sigma_t, noises, labels=None):
        y_pos = (d_alpha_t * target_images + d_sigma_t * noises).flatten(1)
        y_neg, non_nulls = self.sample_negatives(y_pos, labels)
        y_pos = y_pos[non_nulls]
        y_neg = y_neg[non_nulls]
        return y_pos, y_neg
    
    def __call__(self, images, labels):
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        y_pos, y_neg = self.get_pairs(
            target_images=images, 
            d_alpha_t=d_alpha_t, 
            d_sigma_t=d_sigma_t, 
            noises=noises,
            labels=labels,
        )
        sims = F.cosine_similarity(y_pos, y_neg)
        return sims


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args, exp_name): 
    
    dist.init_process_group("nccl")
    rank = dist.get_rank()   
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)

    # if accelerator.is_main_process:
    # if rank == 0:
    os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    save_dir = os.path.join(args.output_dir, exp_name)
    
    os.makedirs(save_dir, exist_ok=True)
    print("Saving to: ", save_dir)
    args_dict = vars(args)
    # Save to a JSON file
    json_dir = os.path.join(save_dir, "args.json")
    with open(json_dir, 'w') as f:
        json.dump(args_dict, f, indent=4)
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(save_dir)
    logger.info(f"Experiment directory created at {save_dir}")
    device = rank % torch.cuda.device_count()
    if args.seed is not None:
        set_seed(args.seed * dist.get_world_size() + rank)

    # create loss function
    velocity_cos_sim_fn = ComputeVelocityCosSims(
        path_type=args.path_type, 
        weighting=args.weighting,
        null_class_idx=args.num_classes
    )
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // dist.get_world_size())
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # if accelerator.is_main_process:
    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # resume:
    # global_step = 0
        
    # progress_bar = tqdm(
    #     range(0, args.max_train_steps),
    #     initial=global_step,
    #     desc="Steps",
    #     # Only show the progress bar once on each machine.
    #     disable=not rank == 0,
    # )

    pbar = train_dataloader
    pbar = tqdm(pbar) if rank == 0 else pbar

    all_sims = []    
    for raw_image, x, y in pbar:
        raw_image = raw_image.to(device)
        x = x.squeeze(dim=1).to(device)
        y = y.to(device)
        
        sims = velocity_cos_sim_fn(x, y).cpu().numpy()
        all_sims.append(sims)
    
    save_file = os.path.join(save_dir, f"rank_{rank}_cosine_sims.npy")
    np.save(save_file, np.concatenate(all_sims))
    if rank == 0:
        logger.info(f"Saved cosine sims to {save_dir}")
    
    if rank == 0:
        logger.info("Done!")
    
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")

    # model
    parser.add_argument("--num-classes", type=int, default=1000)
    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--max-train-steps", type=int, default=400000)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    
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
    
    exp_name = "velocity_sims"
    # print("The experiment name is: ", exp_name)
    try:
        main(args, exp_name)
    except:
        print("Retrying....")
        main(args, exp_name)
    # main(args, exp_name)









