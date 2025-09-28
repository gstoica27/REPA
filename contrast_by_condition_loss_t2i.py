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


def sample_other_context(context, null_indices=None, sample_type="random", atol=1e-3):
    context_flat = torch.nn.functional.normalize(context.flatten(1), dim=-1)
    similarity = context_flat @ context_flat.T
    if sample_type == "random":
        replacement = torch.ones_like(similarity)
    elif sample_type == "similarity":
        replacement = similarity
    elif sample_type == "dissimilarity":
        replacement = 1-similarity
    else:
        raise NotImplementedError("Sample type {} not implemented".format(sample_type))
    
    similarity = torch.where(
        similarity >= (1.0-atol), 
        torch.ones_like(similarity) * -float('inf'), 
        replacement
    )
    similarity.fill_diagonal_(-float('inf'))
    # remove the null token from consideration
    if null_indices is not None:
        similarity[:, null_indices] = -float('inf')
    
    probabilities = F.softmax(similarity, dim=-1)
    # candidates = torch.ones((context.shape[0], context.shape[0]), device=context.device)
    # candidates.fill_diagonal_(-float('inf'))
    # if null_indices is not None:
    #     candidates[:, null_indices] = -float('inf')
    # # Sample random indices based on uniform similarity
    # probabilities = F.softmax(candidates, dim=-1)
    selected_indices = torch.multinomial(probabilities, 1).squeeze(-1)
    selected_context = context[selected_indices]
    # Verify no self-sampling
    check_similarity = F.cosine_similarity(context.flatten(1), selected_context.flatten(1), dim=-1)
    assert (check_similarity < (1.0-atol)).all(), "Self-sampling detected in sample_other_classes"

    return selected_context


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
            contrastive_sampling_type: str = "random"
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
        self.contrastive_sampling_type = contrastive_sampling_type
        print(f"Conditioning contrastive loss on: {self.condition_on}")
        print("Contrastive sampling type: {}".format(self.contrastive_sampling_type))
            
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
    
    def __call__(self, model, images, model_kwargs=None, zs=None, null_token=None):
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
        model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
        positive_loss = mean_flat((model_output - model_target) ** 2)

        if null_token is not None:
            context_flat = F.normalize(model_kwargs['context'].flatten(1), dim=-1)
            null_flat = F.normalize(null_token.flatten(1), dim=-1)
            similarity_to_null = (context_flat @ null_flat.T).squeeze(-1)
            null_indices = (similarity_to_null >= (1.0 - 1e-6))#.nonzero(as_tuple=True)[0]

        if self.condition_on == "context":
            negative_context = sample_other_context(
                context=model_kwargs['context'],
                null_indices=null_indices,
                sample_type=self.contrastive_sampling_type
            )
        elif self.condition_on == "null":
            negative_context = torch.cat([null_token] * model_kwargs['context'].shape[0], dim=0)
        else:
            raise NotImplementedError("Conditioning on {} not implemented".format(self.condition_on))
        
        model_kwargs['context'] = negative_context
        # pdb.set_trace()
        with torch.no_grad():
            neg_output = model(model_input, time_input.flatten(), **model_kwargs)[0]#.detach()
        #     neg_output = 1.0
        elementwise_neg_loss = (model_output - neg_output) ** 2
        if null_token is not None:
            elementwise_neg_loss = elementwise_neg_loss[~null_indices]
        negative_loss = mean_flat(elementwise_neg_loss)
        # pdb.set_trace()
        total_loss = positive_loss.mean() - self.temperature * negative_loss.mean()

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
