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


def triplet_mse_loss(x, y, temperature=1.0, cls=None, choices=None):
    x = x.flatten(1)
    y = y.flatten(1)
    error = ((x[None] - y[:, None]) ** 2).mean(-1)
    indices = torch.arange(x.shape[0]).to(x.device)
    if choices is None:
        choices = torch.tensor([indices[indices != i][torch.randperm(x.shape[0]-1)[0]] for i in range(x.shape[0])])
    assert ((choices == indices.cpu()).sum() == 0).item(), "Triplet loss choices are incorrect"
    negatives = error[np.arange(x.shape[0]), choices]
    positives = error.diagonal()
    loss = positives - temperature * negatives
    return loss


def class_conditioned_triplet_mse_loss(x, y, cls=None, temperature=1.0):
    x, y = x.flatten(1), y.flatten(1)
    error = ((x[None] - y[:, None]) ** 2).mean(-1)
    # Add a column of zeros in case the entire batch contains 1 class
    indices = torch.arange(x.shape[0]).to(y.device)
    error = torch.cat([error, torch.zeros_like(error[:, 1])[:, None]], dim=1)
    choices = []
    pdb.set_trace()
    for idx in range(x.shape[0]):
        nonidx_mask = indices != idx
        nony_mask = cls != cls[idx]
        # if nony_mask.sum().item() < 255:
        #     pdb.set_trace()
        mask = nonidx_mask * nony_mask
        candidates = indices[mask]
        if len(candidates) == 0:
            pdb.set_trace()
            candidate = len(x.shape[0]) 
        else:
            candidate = candidates[torch.randperm(len(candidates))[0]]
        choices.append(candidate)
    choices = torch.tensor(choices).to(x.device).to(torch.int)
    negatives = error[np.arange(x.shape[0]), choices]
    positives = error.diagonal()
    loss = positives - temperature * negatives
    return loss


def choose_denoising_loss(name):
    if name == "mse":
        print('Using MSE loss')
        return mse_flat
    elif name == 'triplet_mse':
        print('Using triplet MSE loss')
        return triplet_mse_loss
    elif name == 'class_conditioned_triplet_mse':
        print('Using class-conditioned triplet MSE loss')
        return class_conditioned_triplet_mse_loss
    else:
        raise NotImplementedError("Denoising loss {} not implemented.".format(name))


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


class TripletSILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            denoising_type="mean",
            denoising_weight=1.0,
            null_class_idx=None,
            dont_contrast_on_unconditional=False,
            is_class_conditioned=False,
            weigh_on_time=False,
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
        self.weigh_on_time = weigh_on_time
        if not self.dont_contrast_on_unconditional:
            assert self.null_class_idx is not None, "Null class index must be provided"
        
        if denoising_type == 'triplet_any_noise':
            print('Using triplet any noise loss')
            self.denoising_fn = self.triplet_any_noise
        elif denoising_type == 'triplet_same_noise':
            print('Using triplet same noise loss')
            self.denoising_fn = self.triplet_same_noise
        elif denoising_type == 'triplet_target_conditioned':
            print('Using triplet target conditioned loss')
            self.denoising_fn = self.compute_target_conditioned_triplet_loss
        
        if self.is_class_conditioned:
            print("Using class-conditioned triplet loss")
        else:
            print("Using standard triplet loss")
            
        self.temperature = denoising_weight
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
    
    def compute_triplet_loss_efficiently(self, x, y, labels=None):
        x = x.flatten(1)
        y = y.flatten(1)
        # Obtain positive samples and compute error
        y_pos = y
        pos_error = mean_flat((x - y_pos) ** 2)
        bsz = x.shape[0]
        choices = torch.tile(torch.arange(bsz), (bsz, 1)).to(x.device)
        choices.fill_diagonal_(-1.)
        choices = choices.sort(dim=1)[0][:, 1:]
        choices = choices[torch.arange(bsz), torch.randint(0, bsz-1, (bsz,))]
        # assert ((choices == torch.arange(bsz).to(x.device)).sum() == 0).item(), "Triplet loss choices are incorrect"
        y_neg = y[choices]
        # Compute error
        if self.dont_contrast_on_unconditional:
            non_nulls = labels != self.null_class_idx
        else:
            non_nulls = torch.ones_like(labels, dtype=torch.bool)
        neg_elem_error = ((x - y_neg) ** 2) * non_nulls.to(x.device).unsqueeze(-1)
        neg_elem_error = neg_elem_error
        neg_error = mean_flat(neg_elem_error) * bsz / non_nulls.sum() # rescale to account for null classes
        # pdb.set_trace()
        # Compute loss
        loss = pos_error - self.temperature * neg_error
        # return loss
        return {
            "loss": loss,
            "flow_loss": pos_error,
            "contrastive_loss": neg_error
        }
    
    def compute_class_conditioned_triplet_loss(self, x, y, labels=None):
        negative_idxs = class_conditioned_sampling(labels)
        bsz = x.shape[0]
        x = x.flatten(1)
        y = y.flatten(1)
        # Obtain positive samples and compute error
        y_pos = y
        pos_error = mean_flat((x - y_pos) ** 2)
        y_neg = y[negative_idxs]
        if self.dont_contrast_on_unconditional:
            non_nulls = labels != self.null_class_idx
        else:
            non_nulls = torch.ones_like(labels, dtype=torch.bool)
        neg_elem_error = ((x - y_neg) ** 2) * non_nulls.to(x.device).unsqueeze(-1)
        neg_error = mean_flat(neg_elem_error) * bsz / non_nulls.sum() # rescale to account for null classes
        # Compute loss
        loss = pos_error - self.temperature * neg_error
        return {
            "loss": loss,
            "flow_loss": pos_error,
            "contrastive_loss": neg_error
        }
    
    def sample_negatives_unconditionally(self, candidates):
        bsz = candidates.shape[0]
        choices = torch.tile(torch.arange(bsz), (bsz, 1)).to(candidates.device)
        choices.fill_diagonal_(-1.)
        choices = choices.sort(dim=1)[0][:, 1:]
        choices = choices[torch.arange(bsz), torch.randint(0, bsz-1, (bsz,))]
        return choices
    
    def sample_negatives_conditionally(self, candidates):
        return class_conditioned_sampling(candidates)
    
    def compute_target_conditioned_triplet_loss(
        self, pred, target_images, d_alpha_t, d_sigma_t, 
        noises, noised_images, time, labels=None, **kwargs
    ):
        # pdb.set_trace()
        # Compute the target trajectory
        model_target = d_alpha_t * target_images + d_sigma_t * noises
        # Compute the contrastive trajectory
        # Get sampling indices
        if self.is_class_conditioned:
            choices = self.sample_negatives_conditionally(labels)
        else:
            choices = self.sample_negatives_unconditionally(labels)
        # pdb.set_trace()
        # Get negative targets
        negative_targets = target_images[choices]
        # Compute negative trajectory
        # negative_model_target = (1 / time) * (noised_images - negative_targets)
        negative_model_target = (noised_images - negative_targets)
        # Compute weights based on whether to contrast on unconditional
        if self.dont_contrast_on_unconditional:
            non_nulls = labels != self.null_class_idx
        else:
            non_nulls = torch.ones_like(labels, dtype=torch.bool)
        # pdb.set_trace()
        # Compute the loss
        pos_loss = mean_flat((pred - model_target) ** 2)
        # neg_loss = mean_flat((pred - negative_model_target) ** 2)
        pred_normalized = F.normalize(pred.flatten(1), dim=-1)
        negative_normalized = F.normalize(negative_model_target.flatten(1), dim=-1)
        # Compute the contrastive loss
        neg_elem_error = ((pred_normalized - negative_normalized) ** 2) * non_nulls.to(pred.device).unsqueeze(-1)
        pdb.set_trace()
        if self.weigh_on_time:
            neg_elem_error = neg_elem_error * time
        neg_loss = - mean_flat(neg_elem_error) * pred.shape[0] / non_nulls.sum() # rescale to account for null classes
        # Compute the final loss
        loss = pos_loss + self.temperature * neg_loss
        return {
            "loss": loss,
            "flow_loss": pos_loss,
            "contrastive_loss": neg_loss
        }
        
    def triplet_any_noise(self, pred, target_images, d_alpha_t, d_sigma_t, noises, labels=None, **kwargs):
        model_target = d_alpha_t * target_images + d_sigma_t * noises
        if self.is_class_conditioned:
            loss = self.compute_class_conditioned_triplet_loss(pred, model_target, labels)
        else:
            loss = self.compute_triplet_loss_efficiently(pred, model_target, labels)
        return loss
    
    def triplet_same_noise(self, pred, target_images, d_alpha_t, d_sigma_t, noises, labels=None):
        reconstructed_target = (pred - d_sigma_t * noises) / d_alpha_t
        loss = self.compute_triplet_loss(reconstructed_target, target_images, labels)
        return loss

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
            
        model_input = alpha_t * images + sigma_t * noises
        model_output, zs_tilde, labels = model(model_input, time_input.flatten(), **model_kwargs)
        # pdb.set_trace()
        denoising_loss = self.denoising_fn(
            pred=model_output, 
            target_images=images, 
            d_alpha_t=d_alpha_t, 
            d_sigma_t=d_sigma_t, 
            noises=noises,
            labels=labels,
            noised_images=model_input,
            time=time_input, 
        )
        # denoising_loss = self.denoising_fn(model_output, model_target, temperature=self.denoising_weight, cls=model_kwargs['y'])
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
