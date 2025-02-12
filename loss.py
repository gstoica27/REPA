import pdb
import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x, temperature=1.0):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x, temperature=1.):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

def mse_flat(x, y, temperature=1):
    err = (x - y) ** 2
    return mean_flat(err)

def self_weighted_mean_flat(x, y, temperature=100):
    """"
    Compute a self-weighted mean over all non-batch dimensions.
    Rather than weighting each element equally, we weight each element according to its loss. 
    This is intended to act as a continuous version of the "disjoint mean" operation.
    """
    err = (x - y) ** 2
    weight = F.tanh(err * temperature)
    z = sum_flat(weight * z)
    denom = sum_flat(weight)
    return z / denom

def contrastive_mse_loss(x, y, temperature=1.):
    x = x.flatten(1)
    y = y.flatten(1)
    contrastive_err = ((x[None] - y[:, None]) ** 2).mean(-1)
    weights = -torch.ones_like(contrastive_err) / (contrastive_err.shape[0] - 1)
    weights.fill_diagonal_(temperature)
    loss = (weights * contrastive_err).sum(-1) / 2.
    return loss

def constrastive_l2_loss(x, y, temperature=1.):
    x = x.flatten(1)
    y = y.flatten(1)
    contrastive_err = ((x[None] - y[:, None]) ** 2).mean(-1) / temperature
    labels = torch.arange(x.shape[0]).to(x.device)
    loss_gen = F.cross_entropy(-contrastive_err, labels)
    loss_rec = F.cross_entropy(-contrastive_err.T, labels)
    loss = (loss_gen + loss_rec) / 2.
    return loss

def contrastive_cos_loss(x, y, temperature=0.):
    x = F.normalize(x.flatten(1), p=2, dim=1)
    y = F.normalize(y.flatten(1), p=2, dim=1)
    contrastive_err = (x @ y.T) / temperature
    labels = torch.arange(x.shape[0]).to(x.device)
    loss_gen = F.cross_entropy(contrastive_err, labels)
    loss_rec = F.cross_entropy(contrastive_err.T, labels)
    loss = (loss_gen + loss_rec) / 2.
    return loss
    
def softmax_weighted_mean(x, temperature=1):
    """
    Compute a weighted mean using softmax weights.
    """
    weight = torch.exp(x * temperature)  # Softmax numerator
    weight = weight / weight.sum()  # Normalize to sum to 1
    z = (weight * x).sum()
    return z

def softmax_weighted_mean(x, temperature=1.0):
    """
    Compute a softmax-weighted mean over all non-batch dimensions.
    """
    z = x.flatten(1)
    weight = F.softmax(z * temperature, dim=-1)
    return sum_flat(weight * z)

def choose_denoising_loss(name):
    if name == "mse":
        print('Using MSE loss')
        return mse_flat
    elif name == "self_weighted_mean":
        print('Using self-weighted mean loss')
        return self_weighted_mean_flat
    elif name == "softmax_weighted_mean":
        print('Using softmax-weighted mean loss')
        return softmax_weighted_mean
    elif name == 'contrastive_l2':
        print('Using contrastive L2 loss')
        return constrastive_l2_loss
    elif name == 'contrastive_mse':
        print('Using contrastive MSE loss')
        return contrastive_mse_loss
    elif name == 'contrastive_cos':
        print('Using contrastive cosine loss')
        return contrastive_cos_loss
    else:
        raise NotImplementedError("Denoising loss {} not implemented.".format(name))

def compute_hsic_parallel(A, B):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) between two sets of features.
    """
    N = A.shape[1]
    hsic_AB = (A.flatten(1, 2).to(torch.float64) @ B.flatten(1, 2).T.to(torch.float64))#.to(torch.float32) # [L1,L2] --good
    A_red = A.to(torch.float64).flatten(1,2).sum(dim=-1, keepdim=True)# reduce_mat(A) # [L1,1]
    B_red = B.to(torch.float64).flatten(1,2).sum(dim=-1, keepdim=True) # reduce_mat(B) # [L2,1]
    hsic_AB += ((A_red @ B_red.T) / ((N - 1) * (N - 2)))#.to(torch.float32)
    
    oneA = A.to(torch.float64).sum(dim=-2) # torch.einsum('ij,ijk->ik', ones.T.repeat(A.shape[0], 1), A)
    Bone = B.to(torch.float64).sum(dim=-1) # B @ ones
    hsic_AB -= (((oneA @ Bone.T) * 2 / (N -2)))#.to(torch.float32)
    return (1 / (N * (N - 3)) * hsic_AB).to(torch.float32)#.item()#.cpu()

def compute_all_hsic_parallel(A, B, denom=1):
    hsic_AB = compute_hsic_parallel(A, B) / denom
    hsic_AA = (compute_hsic_parallel(A, A) / denom).diagonal()[...,None] # [L1, 1]
    hsic_BB = (compute_hsic_parallel(B, B) / denom).diagonal()[None] # [1, L2]
    for hsic in [hsic_AB, hsic_AA, hsic_BB]:
        if torch.isnan(hsic).any(): pdb.set_trace()
        # if (hsic < 0).sum() > 0: pdb.set_trace()
    
    return (
        hsic_AB, 
        hsic_AA.repeat(1, hsic_AB.shape[1]), 
        hsic_BB.repeat(hsic_AB.shape[0], 1)
    )

def compute_cka(feats):
    """
    Compute the Centered Kernel Alignment (CKA) between two sets of features.
    """
    features = torch.stack(feats, dim=0)
    AB, AA, BB = compute_all_hsic_parallel(features, features)
    # return F.relu(AB / (AA.sqrt() * BB.sqrt()))
    return AB / (AA.sqrt() * BB.sqrt())

def compute_gram_matrix(X, method='between_images'):
    """
    Compute the Gram matrix of a set of features.
    """
    assert len(X.shape) == 3, "Input must be of shape [batch, tokens, features]. It currently has shape: {}".format(X.shape)
    
    if method == 'between_images':
        Y = X.flatten(1, 2)                     # [B,TxE]
        gram = torch.mm(Y, Y.T)[None]           # [B,B] -> [1,B,B]
    elif method == 'between_tokens':
        gram = torch.bmm(X, X.transpose(1, 2))  # [B,T,E]x[B,E,T] -> [B,T,T]
    elif method == 'between_images_per_token':
        Y = X.transpose(0, 1)                   # [B,T,E] -> [T,B,E]
        gram = torch.bmm(Y, Y.transpose(1, 2))  # [T,B,E]x[T,E,B] -> [T,B,B]
    else:
        raise NotImplementedError("Method {} not implemented.".format(method))
    # fill in diagonals with 0
    # gram.fill_diagonal_(0)
    gram[:, torch.arange(gram.shape[1]), torch.arange(gram.shape[2])] = 0
    return gram

def compute_hsic(A, B):
    N = A.shape[0]
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    hsic_AB = torch.dot(A.flatten(), B.flatten())
    A_red = A.sum() # [L1,1]
    B_red = B.sum() # [L2,1]
    hsic_AB += ((A_red * B_red) / ((N - 1) * (N - 2)))

    oneA = A.sum(dim=0) 
    Bone = B.sum(dim=-1)
    oneABone = torch.dot(oneA, Bone)
    hsic_AB -= ((oneABone * 2 / (N - 2)))
    return (1 / (N * (N - 3)) * hsic_AB).to(torch.float32)

def compute_batched_hsic(A, B):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) between two sets of features.
    Assume A,B are both of shape [*,L1,L1]
    """
    N = A.shape[1]
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    # hsic_AB = torch.dot(A.flatten(), B.flatten())
    # hsic_AB_ = torch.diagonal(torch.bmm(A, B), dim1=-2, dim2=-1).sum(-1)
    hsic_AB = (A * B).sum((1,2)) # [*,L1,L1] -> [*,1]
    A_red = A.sum((1,2)) # [*,1]
    B_red = B.sum((1,2)) # [*,1]
    hsic_AB += ((A_red * B_red) / ((N - 1) * (N - 2)))
    oneA = A.sum(dim=1) 
    Bone = B.sum(dim=2)
    # oneABone = torch.dot(oneA, Bone)
    oneABone = (oneA * Bone).sum(1)
    hsic_AB -= ((oneABone * 2 / (N - 2)))
    return (1 / (N * (N - 3)) * hsic_AB).to(torch.float32)

def check_for_correct_hsic(A, B):
    Y = [compute_hsic(A[i], B[i]) for i in range(A.shape[0])]
    Y = torch.tensor(Y, device=Y[0].device)
    return Y
    
def compute_pairwise_cka(A, B):
    AB = compute_batched_hsic(A, B)
    AA = compute_batched_hsic(A, A)
    BB = compute_batched_hsic(B, B)
    # # # Check for correctness
    # AB_ = check_for_correct_hsic(A, B)
    # AA_ = check_for_correct_hsic(A, A)
    # BB_ = check_for_correct_hsic(B, B)
    # if not torch.allclose(AB, AB_) or not torch.allclose(AA, AA_) or not torch.allclose(BB, BB_):
    #     print("HSIC computation error")
    
    cka = AB / (AA.sqrt() * BB.sqrt())
    return cka

def compute_cka_loss(cka, add_relu=True):
    if add_relu:
        cka = F.relu(cka)
    return (1 - cka).mean()

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            struct_method=None,
            struct_add_relu=True,
            denoising_type="mean",
            denoising_weight=1.0,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.struct_method = struct_method
        self.struct_add_relu = struct_add_relu
        
        self.denoising_fn = choose_denoising_loss(denoising_type)
        self.denoising_weight = denoising_weight

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
    
    def compute_structure_loss(self, pve_latents, diffusion_latents):
        loss = 0
        # pdb.set_trace()
        for i, (pve_layer_latents, diffusion_layer_latents) in enumerate(zip(pve_latents, diffusion_latents)):
            pve_gram = compute_gram_matrix(pve_layer_latents, method=self.struct_method)
            diffusion_gram = compute_gram_matrix(diffusion_layer_latents, method=self.struct_method)
            cka = compute_pairwise_cka(diffusion_gram, pve_gram)
            cka_loss = compute_cka_loss(cka, self.struct_add_relu)
            loss += cka_loss
        return loss / len(pve_latents)

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
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde, hs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = self.denoising_fn(model_output, model_target, temperature=self.denoising_weight)
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
        # CKA loss
        struct_loss = self.compute_structure_loss(
            pve_latents=zs, diffusion_latents=hs_tilde
        )
        return denoising_loss, proj_loss, struct_loss
