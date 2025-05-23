import pdb
import torch
import numpy as np
from copy import deepcopy


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def detach_and_dtype(x, dtype=torch.float32):
    return x.detach().to(dtype=dtype)


def debias_via_convex_combination(
        velocity, bias, bias_lambda, subtract_bias=False, is_baseline=False
):
    # pdb.set_trace()
    if is_baseline and subtract_bias:
        new_velocity = (velocity + bias_lambda * bias) / (1 - bias_lambda)
    elif is_baseline and not subtract_bias:
        new_velocity = (velocity - bias_lambda * bias) / (1 - bias_lambda)
    elif not is_baseline and subtract_bias:
        new_velocity = (1 - bias_lambda) * velocity - bias_lambda * bias
    elif not is_baseline and not subtract_bias:
        new_velocity = (1 - bias_lambda) * velocity + bias_lambda * bias
    else:
        raise NotImplementedError("is_baseline and subtract_bias not implemented")
    return new_velocity


def debias_via_weighted_combination(
    velocity, bias, bias_lambda, velocity_lambda, subtract_bias=False, is_baseline=False
):
    # pdb.set_trace()
    if is_baseline and subtract_bias:
        new_velocity = (velocity + bias_lambda * bias) / velocity_lambda
    elif is_baseline and not subtract_bias:
        new_velocity = (velocity - bias_lambda * bias) / velocity_lambda
    elif not is_baseline and subtract_bias:
        new_velocity = velocity_lambda * velocity - bias_lambda * bias
    elif not is_baseline and not subtract_bias:
        new_velocity = velocity_lambda * velocity + bias_lambda * bias
    else:
        raise NotImplementedError("is_baseline and subtract_bias not implemented")
    return new_velocity


def debias_via_orthogonal_projection(
    velocity, bias, bias_lambda, velocity_lambda=None, subtract_bias=False, is_baseline=False
):
    # pdb.set_trace()
    if velocity_lambda is None:
        velocity_lambda = 1 - bias_lambda
    debiased_velocity = debias_via_weighted_combination(
        velocity, bias, bias_lambda, velocity_lambda, subtract_bias=subtract_bias, is_baseline=is_baseline
    )
    shape = velocity.shape
    velocity_cond, velocity_uncond = debiased_velocity.chunk(2)
    bias_onto_uncond = (
        (
            (bias * velocity_uncond).sum((1,2,3), keepdim=True) / (velocity_uncond * velocity_uncond).sum((1,2,3), keepdim=True)
        ) * velocity_uncond
    )
    bias_orth_uncond = (bias - bias_onto_uncond) * bias_lambda
    if subtract_bias:
        new_uncond = velocity_uncond - bias_onto_uncond
    else:
        new_uncond = velocity_uncond + bias_onto_uncond
    new_velocity = torch.cat([velocity_cond, new_uncond], dim=0)
    return new_velocity


def orthogonalize_unconditional(velocity, bias_lambda=0.05, velocity_lambda=0.95):
    # pdb.set_trace()
    velocity_cond, velocity_uncond = velocity.chunk(2)
    unconditional_projection = (
        (
            (velocity_cond * velocity_uncond).sum((1,2,3), keepdim=True) / (velocity_cond * velocity_cond).sum((1,2,3), keepdim=True)
        ) * velocity_cond
    )
    velocity_orth_uncond = (velocity_uncond - unconditional_projection)
    velocity_uncond = velocity_lambda * velocity_orth_uncond + bias_lambda * unconditional_projection
    new_velocity = torch.cat([velocity_cond, velocity_uncond], dim=0)
    return new_velocity


def debias_velocity(velocity, bias, bias_lambda, method=None, velocity_lambda=None, subtract_bias=False, is_baseline=False):
    # pdb.set_trace()
    if method is None:
        return velocity
    elif method == 'convex':
        new_velocity = debias_via_convex_combination(
            velocity=velocity, bias=bias, bias_lambda=bias_lambda,
            subtract_bias=subtract_bias, is_baseline=is_baseline
        )
    elif method == 'weighted':
        new_velocity = debias_via_weighted_combination(
            velocity=velocity, bias=bias, bias_lambda=bias_lambda,
            velocity_lambda=velocity_lambda, subtract_bias=subtract_bias,
            is_baseline=is_baseline
        )
    elif method == 'weighted_orthogonal':
        new_velocity = debias_via_orthogonal_projection(
            velocity=velocity, bias=bias, bias_lambda=bias_lambda,
            velocity_lambda=velocity_lambda, subtract_bias=subtract_bias,
            is_baseline=is_baseline
        )
    elif method == 'orthogonal_unconditional':
        new_velocity = orthogonalize_unconditional(
            velocity=velocity, bias_lambda=bias_lambda, 
            velocity_lambda=velocity_lambda
        )
    else:
        raise NotImplementedError("Debiasing method not implemented")
    return new_velocity


def euler_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        record_intermediate_steps=False,
        record_intermediate_steps_freq=None,
        record_trajectory_structure=False,
        trajectory_structure_type=None,
        **kwargs
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    if record_intermediate_steps:
        assert record_intermediate_steps_freq is not None
        intermediates = []
    
    if record_trajectory_structure:
        assert trajectory_structure_type is not None
        x_source = deepcopy(x_next.detach())
        trajectory_vectors = []
    
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y            
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            
            # if bias is not None:
            #     d_cur = (1 - bias_weight) * d_cur + bias_weight * bias

            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)                
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and (i < num_steps - 1):
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    model_input = torch.cat([x_next] * 2)
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    model_input = x_next
                    y_cur = y
                kwargs = dict(y=y_cur)
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    )[0].to(torch.float64)
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
                
            if record_intermediate_steps and (i + 1) % record_intermediate_steps_freq == 0:
                intermediates.append(deepcopy(x_next.detach()).to(torch.float32))
            
            if record_trajectory_structure:
                if trajectory_structure_type == "segment_cosine":
                    trajectory_vectors += [detach_and_dtype(x_next - x_cur)]
                elif trajectory_structure_type == "source_cosine":
                    trajectory_vectors += [detach_and_dtype(x_next - x_source)]
    
    return_dict = {"samples": x_next}
    
    # last step
    if record_intermediate_steps:
        intermediates.append(deepcopy(x_next.detach()).to(torch.float32))
        return_dict["intermediate_steps"] = intermediates
        # return x_next.to(torch.float32), intermediates
    if record_trajectory_structure:
        pdb.set_trace()
        return_dict["trajectory_vectors"] = torch.stack(trajectory_vectors)
    
    return return_dict


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        record_intermediate_steps=False,
        record_intermediate_steps_freq=None,
        record_trajectory_structure=False,
        trajectory_structure_type=None,
        bias=None,
        bias_lambda=0.0,
        subtract_bias=False,
        is_baseline=False,
        velocity_lambda=None,
        debias_method=None,
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    if record_intermediate_steps:
        assert record_intermediate_steps_freq is not None
        intermediates = []
        expecteds = []
    
    if record_trajectory_structure:
        assert trajectory_structure_type is not None
        x_source = deepcopy(x_next.detach())
        trajectory_vectors = []
    
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y            
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)            
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)

            # if bias is not None:
                # if not is_baseline:
                #     if subtract_bias:
                #         v_cur = (1 - bias_lambda) * v_cur - bias_lambda * bias
                #     else:
                #         v_cur = (1 - bias_lambda) * v_cur + bias_lambda * bias
                # else:
                #     if subtract_bias:
                #         v_cur = (v_cur + bias_lambda * bias) / (1 - bias_lambda)
                #     else:
                #         v_cur = (v_cur - bias_lambda * bias) / (1 - bias_lambda)
            if t_cur <= guidance_high and t_cur >= guidance_low:
                v_cur = debias_velocity(
                    velocity=v_cur, bias=bias, bias_lambda=bias_lambda, 
                    subtract_bias=subtract_bias, is_baseline=is_baseline, 
                    velocity_lambda=velocity_lambda, method=debias_method
                )
            
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
            x_final = x_cur + d_cur * (t_steps[-1] - t_cur)
            
            if record_intermediate_steps and (i + 1) % record_intermediate_steps_freq == 0:
                intermediates.append(deepcopy(x_next.detach().to(torch.float32)))
                '''
                expecteds += [
                    deepcopy(
                        # (latents - v_cur).detach().to(torch.float32)
                        (x_cur + d_cur * torch.sqrt(diffusion) * deps).detach().to(torch.float32)
                    )
                ]
                '''
                expecteds += [x_final.detach().to(torch.float32)]
            
            if record_trajectory_structure:
                if trajectory_structure_type == "segment_cosine":
                    trajectory_vectors += [detach_and_dtype(x_next - x_cur)]
                elif trajectory_structure_type == "source_cosine":
                    trajectory_vectors += [detach_and_dtype(x_next - x_source)]
                elif trajectory_structure_type == "straightness":
                    trajectory_vectors += [detach_and_dtype(x_next - x_cur)]
                elif trajectory_structure_type == "length":
                    trajectory_vectors += [detach_and_dtype(x_next - x_cur)]
                else:
                    raise NotImplementedError("trajectory structure type not implemented")

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        )[0].to(torch.float64)
    
    # if bias is not None:
        # if not is_baseline:
        #     if subtract_bias:
        #         v_cur = (1 - bias_lambda) * v_cur - bias_lambda * bias
        #     else:
        #         v_cur = (1 - bias_lambda) * v_cur + bias_lambda * bias
        # else:
        #     if subtract_bias:
        #         v_cur = (v_cur + bias_lambda * bias) / (1 - bias_lambda)
        #     else:
        #         v_cur = (v_cur - bias_lambda * bias) / (1 - bias_lambda)
    if t_cur <= guidance_high and t_cur >= guidance_low:
        v_cur = debias_velocity(
            velocity=v_cur, bias=bias, bias_lambda=bias_lambda, 
            subtract_bias=subtract_bias, is_baseline=is_baseline, 
            velocity_lambda=velocity_lambda, method=debias_method
        )

    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
    
    return_dict = {"samples": mean_x}
    
    if record_intermediate_steps:
        intermediates.append(deepcopy(mean_x.detach().to(torch.float32)))
        return_dict["intermediate_steps"] = intermediates
        return_dict["expecteds"] = expecteds
        # return mean_x.to(torch.float32), intermediates
    
    if record_trajectory_structure:
        if trajectory_structure_type == "segment_cosine":
            trajectory_vectors += [detach_and_dtype(mean_x - x_cur)]
        elif trajectory_structure_type == "source_cosine":
            trajectory_vectors += [detach_and_dtype(mean_x - x_source)]
        elif trajectory_structure_type == "straightness":
            trajectory_vectors += [detach_and_dtype(mean_x - x_cur)]
        elif trajectory_structure_type == "length":
            trajectory_vectors += [detach_and_dtype(mean_x - x_cur)]
        else:
            raise NotImplementedError("trajectory structure type not implemented")
        
        return_dict["trajectory_vectors"] = torch.stack(trajectory_vectors)
    
    # return mean_x
    return return_dict
