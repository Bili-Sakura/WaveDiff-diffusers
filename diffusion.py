import numpy as np
import torch
from diffusers import DDPMScheduler


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * \
        (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def get_beta_schedule(args):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)

    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.to(dtype=torch.float32)


def build_scheduler(args, device):
    betas = get_beta_schedule(args)
    scheduler = DDPMScheduler(
        num_train_timesteps=betas.shape[0],
        trained_betas=betas.cpu().numpy(),
        prediction_type="sample",
        clip_sample=False,
    )
    return scheduler.to(device)


def add_noise_pair(scheduler, x_start, timesteps, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    timesteps = timesteps.to(x_start.device)
    t_plus_one = timesteps + 1
    x_t = scheduler.add_noise(x_start, noise, timesteps)
    x_t_plus_one = scheduler.add_noise(x_start, noise, t_plus_one)
    return x_t, x_t_plus_one


def scheduler_step(scheduler, model_output, timesteps, sample):
    if torch.is_tensor(timesteps) and timesteps.ndim > 0:
        prev_sample = torch.empty_like(sample)
        for timestep in torch.unique(timesteps):
            t_value = int(timestep)
            mask = timesteps == timestep
            step_output = scheduler.step(
                model_output[mask], t_value, sample[mask])
            prev_sample[mask] = step_output.prev_sample
        return prev_sample

    t_value = int(timesteps) if torch.is_tensor(timesteps) else int(timesteps)
    return scheduler.step(model_output, t_value, sample).prev_sample


def sample_from_model(scheduler, generator, n_time, x_init, opt):
    scheduler.set_timesteps(n_time, device=x_init.device)
    x = x_init
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            t_value = int(timestep) if torch.is_tensor(timestep) else int(timestep)
            t_batch = torch.full(
                (x.size(0),), t_value, dtype=torch.int64, device=x.device)
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_batch, latent_z)
            x = scheduler.step(x_0, t_value, x).prev_sample.detach()
    return x
