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


def sample_other_classes(labels):
    bsz = labels.shape[0]
    num_classes = labels.max() + 1
    random_labels = torch.randint(0, num_classes, (bsz,), device=labels.device)
    mask = random_labels == labels
    while mask.any():
        random_labels[mask] = torch.randint(0, num_classes, (mask.sum(),), device=labels.device)
        mask = random_labels == labels
    assert (random_labels != labels).all(), "Self-sampling detected in sample_other_classes"
    return random_labels


class ContrastByCondition:
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
            condition_on: str = "class",
            detached_component: str = "contraster",
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.null_class_idx = null_class_idx
        self.condition_on = condition_on
        self.detached_component = detached_component
        print(f"Conditioning contrastive loss on: {self.condition_on}")
        if self.condition_on == "null" and self.null_class_idx is None:
            raise ValueError("null_class_idx must be specified when conditioning on null.")
        print(f"Detached component: {self.detached_component}")
            
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
        # pdb.set_trace()
        if self.condition_on == "class":
            negative_labels = sample_other_classes(labels)
        elif self.condition_on == "null":
            negative_labels = torch.full_like(labels, self.null_class_idx)
        else:
            raise NotImplementedError("Conditioning on {} not implemented".format(self.condition_on))
        
        model_kwargs['y'] = negative_labels
        # pdb.set_trace()
        if self.detached_component != "none":
            if self.detached_component == "contraster":
                with torch.no_grad():
                    neg_output = model(model_input, time_input.flatten(), dont_drop=True, **model_kwargs)[0]
                model_output_ = model_output
            elif self.detached_component == "contrasted":
                neg_output = model(model_input, time_input.flatten(), dont_drop=True, **model_kwargs)[0]
                model_output_ = model_output.detach()
            else:
                raise NotImplementedError("Detached component {} not implemented".format(self.detached_component))
                
            #     neg_output = 1.0
            
            elementwise_neg_loss = (model_output_ - neg_output) ** 2
            if self.null_class_idx is not None:
                elementwise_neg_loss = elementwise_neg_loss[labels != self.null_class_idx]
            negative_loss = mean_flat(elementwise_neg_loss)
            # pdb.set_trace()
            total_loss = positive_loss.mean() - self.temperature * negative_loss.mean()
        
        else:
            neg_output = model(model_input, time_input.flatten(), dont_drop=True, **model_kwargs)[0]
            altered_target = model_target * (1 + self.temperature) - self.temperature * neg_output
            loss = (model_output - altered_target) ** 2
            loss[labels == self.null_class_idx] = ((model_output - model_target) ** 2)[labels == self.null_class_idx]
            total_loss = mean_flat(loss)
            negative_loss = torch.tensor(0.0, device=total_loss.device)


        denoising_loss = {
            "loss": total_loss,
            "flow_loss": positive_loss,
            "contrastive_loss": negative_loss
        }
        # pdb.set_trace()
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
