"""
Heavily modified from REPA: https://github.com/sihyun-yu/REPA/blob/main/loss.py
"""

import pdb
import torch
import numpy as np
import torch.nn.functional as F


def mean_flat(x, temperature=1.0, **kwargs):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x, temperature=1.):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def mse_flat(x, y, temperature=1, **kwargs):
    err = (x - y) ** 2
    return mean_flat(err)


def class_conditioned_sampling(labels):
    bsz = labels.shape[0]
    mask = ~(labels[None] == labels[:, None])
    # choices[mask] = -1 # remove self-sampling
    # Now randomly sample from the remaining choices
    weights = mask.float()
    weights_sum = weights.sum(dim=1, keepdim=True)
    if (weights_sum == 0).any():
        # In case there are no valid choices, fallback to uniform sampling
        choices = torch.randint(0, bsz, (bsz,), device=labels.device)
        # weights = torch.ones_like(choices).float()
    else:
        # Normalize weights to avoid division by zero
        weights = weights / weights_sum.clamp(min=1)
        # Sample from the available choices based on weights
        choices = torch.multinomial(weights, 1).squeeze(1)
    # Ensure no self-sampling
    assert (choices != torch.arange(bsz, device=labels.device)).all(), "Self-sampling detected in class_conditioned_sampling"
    return choices

def sample_other_classes(labels):
    bsz = labels.shape[0]
    num_classes = labels.max() + 1
    random_labels = torch.randint(0, num_classes, (bsz,), device=labels.device)
    mask = random_labels == labels
    while mask.any():
        random_labels[mask] = torch.randint(0, num_classes, (mask.sum(),), device=labels.device)
        mask = random_labels == labels
    return random_labels


class ContrastByClass:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            contrastive_weight=1.0,
            null_class_idx=None,
            dont_contrast_on_unconditional=False,
            is_class_conditioned=False,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.null_class_idx = null_class_idx
        self.dont_contrast_on_unconditional = dont_contrast_on_unconditional
        self.is_class_conditioned = is_class_conditioned
        if self.dont_contrast_on_unconditional:
            assert self.null_class_idx is not None, "Null class index must be provided"
            
        self.temperature = contrastive_weight
        print(f"Using temperature of: {self.temperature}")

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
    
    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
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
        
        model_target = d_alpha_t * images + d_sigma_t * noises
        model_input = alpha_t * images + sigma_t * noises
        model_output, zs_tilde, labels = model(model_input, time_input.flatten(), **model_kwargs)
        positive_loss = mean_flat((model_output - model_target) ** 2)
        pdb.set_trace()
        negative_labels = sample_other_classes(model_kwargs['y'])
        model_kwargs['y'] = negative_labels
        neg_output, _, _ = model(model_input, time_input.flatten(), **model_kwargs)
        elementwise_neg_loss = (model_output - neg_output.detach()) ** 2
        if self.null_class_idx is not None:
            elementwise_neg_loss = elementwise_neg_loss[labels != self.null_class_idx]
        negative_loss = mean_flat(elementwise_neg_loss)
        total_loss = positive_loss - self.temperature * negative_loss

        denoising_loss = {
            "loss": total_loss,
            "flow_loss": positive_loss,
            "contrastive_loss": negative_loss
        }
        pdb.set_trace()
        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        
        return denoising_loss, proj_loss
