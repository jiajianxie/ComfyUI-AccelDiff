"""
SADA (Stability-guided Adaptive Diffusion Acceleration) - Sampler Only
Implements step skipping based on momentum stability criteria.
Adapted from: https://github.com/sada-icml (ICML 2025)

- Computes a momentum indicator from the denoising trajectory (using ODE derivative)
- When momentum <= 0, the trajectory is "stable" and the next step can be skipped
- Skipped steps are approximated via Adams-Bashforth predictor or Lagrange interpolation

Unified implementation for ComfyUI's k-diffusion framework:
- All models (FLUX, SD, PixArt, etc.) are wrapped to the same interface:
  model(x, sigma) -> denoised, then d = to_d(x, sigma, denoised)
- Momentum criterion: residual_factor * acceleration (3rd-order Adams-Bashforth)
- Skip data reconstruction in x0 space
"""

from scipy import integrate
import torch
from tqdm.auto import trange

from comfy.k_diffusion import utils
from comfy.samplers import KSamplerX0Inpaint, Sampler


class KSAMPLER(Sampler):
    """SADA sampler - step skipping based on momentum stability.
    
    Unified interface matching other AccelDiff samplers (ZEUS, AdaptiveDiff, EasyCache).
    """
    def __init__(self, sampler_function,
                 max_interval=4, acc_start=10, acc_end=47,
                 lagrange_term=3, lagrange_int=4, lagrange_step=20,
                 extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.max_interval = max_interval
        self.acc_start = max(acc_start, 3)
        self.acc_end = acc_end
        self.lagrange_term = lagrange_term
        self.lagrange_int = lagrange_int
        self.lagrange_step = lagrange_step
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False):
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(
            model_k, noise, sigmas,
            max_interval=self.max_interval,
            acc_start=self.acc_start,
            acc_end=self.acc_end,
            lagrange_term=self.lagrange_term,
            lagrange_int=self.lagrange_int,
            lagrange_step=self.lagrange_step,
            extra_args=extra_args, callback=k_callback, disable=disable_pbar,
            **self.extra_options
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples


def lagrange_skip(t_points, x_values, t_eval):
    """Lagrange polynomial interpolation for step skipping."""
    P_of_t = torch.zeros_like(x_values[0])
    n = len(t_points)
    for i in range(n):
        term = x_values[i].clone()
        for j in range(n):
            if j != i:
                factor = (t_eval - t_points[j]) / (t_points[i] - t_points[j])
                term = term * factor
        P_of_t += term
    return P_of_t


# ===================== Utility functions =====================

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def default_noise_sampler(x, seed=None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None
    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


# ===================== Unified SADA Criterion =====================

def _sada_criterion(i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
                    use_lagrange, lagrange_term, lagrange_int, lagrange_step,
                    lagrange_x0_list, lagrange_step_list,
                    x, sigmas, denoised):
    """Unified SADA momentum criterion for all models.
    
    Uses 3rd-order Adams-Bashforth predictor to compute momentum:
    - pred_prev_sample = x + dt * (23/12*f - 16/12*prev_f[-1] + 5/12*prev_f[-2])
      simplified to: x + dt * (0.625*f + 0.75*prev_f[-1] - 0.375*prev_f[-2])
      (Note: these coefficients come from the official SADA paper's formulation)
    - residual_factor = prev_sample - pred_prev_sample
    - acceleration = (f - prev_f[-1]) - (prev_f[-1] - prev_f[-2])
    - momentum = residual_factor * acceleration
    
    When momentum <= 0, the trajectory is stable and can be skipped.
    Skip data is reconstructed in x0 space or via Lagrange interpolation.
    
    Returns: (skip_this_step, cons_skip, pred_m_m_1)
    """
    skip_this_step = False
    pred_m_m_1 = None
    lagrange_this_step = False

    if prev_f[0] is not None:
        dt = sigmas[i + 1] - sigmas[i]
        pred_prev_sample = x + dt * (0.625 * f + 0.75 * prev_f[-1] - 0.375 * prev_f[-2])
        residual_factor = prev_sample - pred_prev_sample
        acceleration = (f - prev_f[-1]) - (prev_f[-1] - prev_f[-2])
        momentum = residual_factor * acceleration

        momentum_mean = momentum.mean()

        if cons_skip >= max_interval:
            skip_this_step = False
            cons_skip = 0
        elif not use_lagrange and (momentum_mean <= 0 and i in range(acc_start, acc_end)):
            skip_this_step = True
            cons_skip += 1
        elif use_lagrange and (momentum_mean <= 0 and i in range(acc_start, lagrange_step)):
            skip_this_step = True
            cons_skip += 1
        elif use_lagrange and (i % lagrange_int != 0 and i in range(lagrange_step, acc_end)):
            skip_this_step = True
            cons_skip += 1
            lagrange_this_step = True
        else:
            skip_this_step = False
            cons_skip = 0

        # Skip data reconstruction
        if skip_this_step:
            if not lagrange_this_step:
                sigma_next = sigmas[i + 1]
                if sigma_next > 0:
                    pred_m_m_1 = pred_prev_sample - utils.append_dims(sigma_next, pred_prev_sample.ndim) * f
                else:
                    pred_m_m_1 = denoised.clone()
            else:
                valid_points = [(s, v) for s, v in zip(lagrange_step_list, lagrange_x0_list) if s is not None and v is not None]
                if len(valid_points) >= 2:
                    steps_t = [p[0] for p in valid_points]
                    vals = [p[1] for p in valid_points]
                    pred_m_m_1 = lagrange_skip(steps_t, vals, i + 1)
                else:
                    pred_m_m_1 = denoised.clone()

    return skip_this_step, cons_skip, pred_m_m_1


# ===================== SADA Sampler Functions =====================

@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                 max_interval=4, acc_start=10, acc_end=47,
                 lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated Euler sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    for i in trange(N, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (N), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigma_hat * s_in, **extra_args)

        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        prev_sample = x + d * dt

        f = d
        if use_lagrange and i % lagrange_int == 1:
            for k in range(lagrange_term - 1):
                lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                lagrange_step_list[k] = lagrange_step_list[k + 1]
            lagrange_x0_list[-1] = denoised
            lagrange_step_list[-1] = i

        skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
            i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
            use_lagrange, lagrange_term, lagrange_int, lagrange_step,
            lagrange_x0_list, lagrange_step_list,
            x, sigmas, denoised,
        )

        prev_f[0] = prev_f[1]
        prev_f[1] = prev_f[2]
        prev_f[2] = f
        x = prev_sample

    return x


@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None,
                s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                max_interval=4, acc_start=10, acc_end=47,
                lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated Heun sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    for i in trange(N, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (N), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigma_hat * s_in, **extra_args)

        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat

        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            prev_sample = x + d_prime * dt

            f = d
            if use_lagrange and i % lagrange_int == 1:
                for k in range(lagrange_term - 1):
                    lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                    lagrange_step_list[k] = lagrange_step_list[k + 1]
                lagrange_x0_list[-1] = denoised
                lagrange_step_list[-1] = i

            skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
                i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
                use_lagrange, lagrange_term, lagrange_int, lagrange_step,
                lagrange_x0_list, lagrange_step_list,
                x, sigmas, denoised,
            )

            prev_f[0] = prev_f[1]
            prev_f[1] = prev_f[2]
            prev_f[2] = f
            x = prev_sample

    return x


@torch.no_grad()
def sample_heunpp2(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                   max_interval=4, acc_start=10, acc_end=47,
                   lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated Heun++ 2nd order sampler."""
    return sample_heun(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                       s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise,
                       max_interval=max_interval, acc_start=acc_start, acc_end=acc_end,
                       lagrange_term=lagrange_term, lagrange_int=lagrange_int,
                       lagrange_step=lagrange_step)


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                 max_interval=4, acc_start=10, acc_end=47,
                 lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated DPM-Solver-2 sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    for i in trange(N, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (N), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigma_hat * s_in, **extra_args)

        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        if sigmas[i + 1] == 0:
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            prev_sample = x + d_2 * dt_2

            f = d
            if use_lagrange and i % lagrange_int == 1:
                for k in range(lagrange_term - 1):
                    lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                    lagrange_step_list[k] = lagrange_step_list[k + 1]
                lagrange_x0_list[-1] = denoised
                lagrange_step_list[-1] = i

            skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
                i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
                use_lagrange, lagrange_term, lagrange_int, lagrange_step,
                lagrange_x0_list, lagrange_step_list,
                x, sigmas, denoised,
            )

            prev_f[0] = prev_f[1]
            prev_f[1] = prev_f[2]
            prev_f[2] = f
            x = prev_sample

    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None,
                           eta=1., s_noise=1., noise_sampler=None,
                           max_interval=4, acc_start=10, acc_end=47,
                           lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated Euler Ancestral sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    for i in trange(N, disable=disable):
        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigmas[i] * s_in, **extra_args)

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        prev_sample = x + d * dt
        if sigmas[i + 1] > 0:
            prev_sample = prev_sample + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        f = d
        if use_lagrange and i % lagrange_int == 1:
            for k in range(lagrange_term - 1):
                lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                lagrange_step_list[k] = lagrange_step_list[k + 1]
            lagrange_x0_list[-1] = denoised
            lagrange_step_list[-1] = i

        skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
            i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
            use_lagrange, lagrange_term, lagrange_int, lagrange_step,
            lagrange_x0_list, lagrange_step_list,
            x, sigmas, denoised,
        )

        prev_f[0] = prev_f[1]
        prev_f[1] = prev_f[2]
        prev_f[2] = f
        x = prev_sample

    return x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None,
                           eta=1., s_noise=1., noise_sampler=None,
                           max_interval=4, acc_start=10, acc_end=47,
                           lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated DPM-2 Ancestral sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    for i in trange(N, disable=disable):
        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigmas[i] * s_in, **extra_args)

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)

        if sigma_down == 0:
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            prev_sample = x + d_2 * dt_2
            prev_sample = prev_sample + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

            f = d
            if use_lagrange and i % lagrange_int == 1:
                for k in range(lagrange_term - 1):
                    lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                    lagrange_step_list[k] = lagrange_step_list[k + 1]
                lagrange_x0_list[-1] = denoised
                lagrange_step_list[-1] = i

            skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
                i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
                use_lagrange, lagrange_term, lagrange_int, lagrange_step,
                lagrange_x0_list, lagrange_step_list,
                x, sigmas, denoised,
            )

            prev_f[0] = prev_f[1]
            prev_f[1] = prev_f[2]
            prev_f[2] = f
            x = prev_sample

    return x


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4,
               max_interval=4, acc_start=10, acc_end=47,
               lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated LMS sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    ds = []
    for i in trange(N, disable=disable):
        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigmas[i] * s_in, **extra_args)

        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        cur_order = min(i + 1, order)
        coeffs = [linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) for j in range(cur_order)]
        prev_sample = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        f = d
        if use_lagrange and i % lagrange_int == 1:
            for k in range(lagrange_term - 1):
                lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                lagrange_step_list[k] = lagrange_step_list[k + 1]
            lagrange_x0_list[-1] = denoised
            lagrange_step_list[-1] = i

        skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
            i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
            use_lagrange, lagrange_term, lagrange_int, lagrange_step,
            lagrange_x0_list, lagrange_step_list,
            x, sigmas, denoised,
        )

        prev_f[0] = prev_f[1]
        prev_f[1] = prev_f[2]
        prev_f[2] = f
        x = prev_sample

    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None,
                    max_interval=4, acc_start=10, acc_end=47,
                    lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated DPM++ 2M sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(N, disable=disable):
        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            prev_sample = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            prev_sample = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised

        d = to_d(x, sigmas[i], denoised)
        f = d
        if use_lagrange and i % lagrange_int == 1:
            for k in range(lagrange_term - 1):
                lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                lagrange_step_list[k] = lagrange_step_list[k + 1]
            lagrange_x0_list[-1] = denoised
            lagrange_step_list[-1] = i

        skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
            i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
            use_lagrange, lagrange_term, lagrange_int, lagrange_step,
            lagrange_x0_list, lagrange_step_list,
            x, sigmas, denoised,
        )

        prev_f[0] = prev_f[1]
        prev_f[1] = prev_f[2]
        prev_f[2] = f
        x = prev_sample

    return x


@torch.no_grad()
def sample_ddim(model, x, sigmas, extra_args=None, callback=None, disable=None,
                max_interval=4, acc_start=10, acc_end=47,
                lagrange_term=3, lagrange_int=4, lagrange_step=20):
    """SADA-accelerated DDIM sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0

    prev_f = [None, None, None]
    skip_this_step = False
    cons_skip = 0
    pred_m_m_1 = None
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    for i in trange(N, disable=disable):
        if skip_this_step and pred_m_m_1 is not None:
            denoised = pred_m_m_1
            skip_this_step = False
        else:
            denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        prev_sample = x + d * dt

        f = d
        if use_lagrange and i % lagrange_int == 1:
            for k in range(lagrange_term - 1):
                lagrange_x0_list[k] = lagrange_x0_list[k + 1]
                lagrange_step_list[k] = lagrange_step_list[k + 1]
            lagrange_x0_list[-1] = denoised
            lagrange_step_list[-1] = i

        skip_this_step, cons_skip, pred_m_m_1 = _sada_criterion(
            i, f, prev_f, prev_sample, cons_skip, max_interval, acc_start, acc_end,
            use_lagrange, lagrange_term, lagrange_int, lagrange_step,
            lagrange_x0_list, lagrange_step_list,
            x, sigmas, denoised,
        )

        prev_f[0] = prev_f[1]
        prev_f[1] = prev_f[2]
        prev_f[2] = f
        x = prev_sample

    return x


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]
