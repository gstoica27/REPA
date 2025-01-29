import pdb
import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


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

def compute_gram_matrix(X, tokens_are_examples=False):
    """
    Compute the Gram matrix of a set of features.
    """
    assert len(X.shape) == 3, "Input must be of shape [batch, tokens, features]. It currently has shape: {}".format(X.shape)
    
    if tokens_are_examples:
        Y = X.flatten(0, 1)
    else:
        Y = X.flatten(1, 2)
    # compute gram matrix
    gram = Y @ Y.T
    # fill in diagonals with 0
    gram.fill_diagonal_(0)
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

def compute_pairwise_cka(A, B):
    AB = compute_hsic(A, B)
    AA = compute_hsic(A, A)
    BB = compute_hsic(B, B)
    cka = AB / (AA.sqrt() * BB.sqrt())
    return cka

def compute_cka_loss(cka):
    return 1 - F.relu(cka)

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
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

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
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

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
        struct_loss = 0
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            z_gram = compute_gram_matrix(z)
            z_tilde_gram = compute_gram_matrix(z_tilde)
            cka = compute_pairwise_cka(z_gram, z_tilde_gram)
            cka_loss = compute_cka_loss(cka)
            struct_loss += cka_loss
            # sanity checking cka computation
            cka_ref = compute_cka([z_gram, z_tilde_gram])[0, -1]
            if not torch.isclose(cka, cka_ref, atol=1e-5):
                print("CKA computation error")
                print(cka.item(), cka_ref.item())
        struct_loss /= len(zs)
        # print("Looking at shapes")
        # print(len(zs), len(zs_tilde))
        # print(zs[0].shape, zs_tilde[0].shape)
        # import pdb; pdb.set_trace()
        # X = compute_gram_matrix(zs[0])
        # Y = compute_gram_matrix(zs_tilde[0])
        # cka_simpler = compute_pairwise_cka(X, Y)
        # cka_og = compute_cka([X, Y])
        return denoising_loss, proj_loss, struct_loss
