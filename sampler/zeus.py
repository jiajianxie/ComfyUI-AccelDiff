"""
ZEUS (Zero-shot Efficient Unified Sparsity) - Sampler Level
Implements modular step-skipping with PSI interpolation and optional Lagrange interpolation.
Adapted from: https://github.com/Ting-Justin-Jiang/ZEUS

Key features:
- Fixed-pattern step skipping: skip steps where (step_index % denominator) in modular
- PSI interpolation: linear extrapolation on ODE derivative (d) for skipped steps
- Reuse-Interp alternating: odd consecutive skips reuse cached interpolation
- Optional Lagrange N-th order polynomial interpolation for later steps
- No model-side patch needed (pure sampler-level acceleration)

IMPORTANT: In ComfyUI's k-diffusion framework, model(x, sigma) returns denoised (x_0).
The ODE derivative d = (x - denoised) / sigma is the quantity analogous to the model's
velocity/epsilon output. PSI interpolation must be done in d-space (not denoised-space)
to match the official ZEUS implementation.
"""

import torch
from tqdm.auto import trange

from comfy.k_diffusion import utils
from comfy.samplers import KSamplerX0Inpaint, Sampler


class KSAMPLER(Sampler):
    def __init__(self, sampler_function,
                 denominator=3, modular=(0, 1),
                 acc_start=10, acc_end=45,
                 interp_mode="psi",
                 caching_mode="reuse_interp",
                 max_interval=4,
                 lagrange_term=3, lagrange_int=6, lagrange_step=24,
                 extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.denominator = denominator
        self.modular = modular
        self.acc_start = max(acc_start, 3)
        self.acc_end = acc_end
        self.interp_mode = interp_mode
        self.caching_mode = caching_mode
        self.max_interval = max_interval
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
            denominator=self.denominator,
            modular=self.modular,
            acc_start=self.acc_start,
            acc_end=self.acc_end,
            interp_mode=self.interp_mode,
            caching_mode=self.caching_mode,
            max_interval=self.max_interval,
            lagrange_term=self.lagrange_term,
            lagrange_int=self.lagrange_int,
            lagrange_step=self.lagrange_step,
            extra_args=extra_args, callback=k_callback, disable=disable_pbar,
            **self.extra_options
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples


# ===================== Utility functions =====================

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)


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


# ===================== Common ZEUS helpers =====================

def _zeus_init(N, acc_start, acc_end, lagrange_term, modular):
    """Common ZEUS state initialization."""
    acc_start = max(acc_start, 3)
    if acc_end < 0:
        acc_end = N + acc_end
    use_lagrange = lagrange_term > 0
    modular_set = set(modular) if not isinstance(modular, set) else modular

    prev_d = [None, None, None]  # ODE derivative history for PSI
    skip_this_step = False
    cons_skip = 0
    prev_interp = None
    lagrange_this_step = False
    lagrange_x0_list = [None] * lagrange_term if use_lagrange else []
    lagrange_step_list = [None] * lagrange_term if use_lagrange else []

    return (acc_start, acc_end, use_lagrange, modular_set,
            prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
            lagrange_x0_list, lagrange_step_list)


def _zeus_should_skip(i, prev_d, cons_skip,
                      denominator, modular, acc_start, acc_end,
                      max_interval,
                      use_lagrange, lagrange_int, lagrange_step):
    """Determine whether the NEXT step should be skipped.
    
    The decision is based on the CURRENT step index `i` (matching the official
    ZEUS implementation where `_step_index` is checked before incrementing).
    If current step `i` satisfies the skip criterion, the NEXT step is skipped.
    
    Returns: (skip_next, cons_skip, lagrange_this_step)
    """
    skip_next = False
    lagrange_this_step = False

    if prev_d[0] is not None:
        if cons_skip >= max_interval:
            skip_next = False
            cons_skip = 0
        elif not use_lagrange and (i % denominator in modular and acc_start <= i < acc_end):
            skip_next = True
            cons_skip += 1
        elif use_lagrange and (i % denominator in modular and acc_start <= i < lagrange_step):
            skip_next = True
            cons_skip += 1
        elif use_lagrange and (i % lagrange_int != 0 and lagrange_step <= i < acc_end):
            skip_next = True
            cons_skip += 1
            lagrange_this_step = True
        else:
            skip_next = False
            cons_skip = 0

    return skip_next, cons_skip, lagrange_this_step


def _zeus_psi_interp_d(prev_d, cons_skip, prev_interp, caching_mode):
    """PSI interpolation in ODE derivative (d) space.
    Returns: (d_interp, prev_interp)
    """
    d_interp = prev_d[-1] + (prev_d[-1] - prev_d[-2])

    if caching_mode == "reuse_interp" and cons_skip % 2 == 1:
        if cons_skip == 1:
            prev_interp = d_interp.clone()
        d_interp = prev_interp
    elif caching_mode == "interp_all" or (caching_mode == "reuse_interp" and cons_skip == 1):
        prev_interp = d_interp.clone()

    return d_interp, prev_interp


def _zeus_update_d_history(prev_d, d, was_skipped=False):
    """Shift d history and append new entry.
    
    When was_skipped=True, we shift the history but clone the latest real value
    (prev_d[-1]) into the new slot, rather than storing the interpolated d.
    This matches the official ZEUS FluxTransformer2DModel forward behavior where
    skipped steps do: shift prev_f, then prev_f[-1] = prev_f[-1].clone().
    This prevents interpolated values from polluting the history while maintaining
    correct shift behavior for consecutive skips.
    """
    if not was_skipped:
        for k in range(2):
            prev_d[k] = prev_d[k + 1]
        prev_d[-1] = d.clone()
    else:
        # Mimic official: shift + clone latest real value
        latest_real = prev_d[-1]
        for k in range(2):
            prev_d[k] = prev_d[k + 1]
        prev_d[-1] = latest_real  # keep the same real value (already a tensor, no need to clone again)


def _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                          lagrange_term, lagrange_x0_list, lagrange_step_list):
    """Update Lagrange anchor points."""
    if use_lagrange and lagrange_int is not None and i % lagrange_int == 1:
        for k in range(lagrange_term - 1):
            lagrange_x0_list[k] = lagrange_x0_list[k + 1]
            lagrange_step_list[k] = lagrange_step_list[k + 1]
        lagrange_x0_list[-1] = denoised.clone()
        lagrange_step_list[-1] = i


def _zeus_handle_skip(x, sigma, s_in, model, extra_args,
                      skip_this_step, lagrange_this_step,
                      prev_d, cons_skip, prev_interp, caching_mode,
                      lagrange_x0_list, lagrange_step_list, i):
    """Handle a potentially skipped step: either interpolate or evaluate model.
    
    Returns: (denoised, d, was_skipped, prev_interp)
    
    IMPORTANT: was_skipped indicates whether this step was skipped. When True,
    the caller must NOT update prev_d history with the returned d, because it's
    an interpolated value, not a real model output. The official ZEUS implementation
    keeps prev_f history clean (only real model outputs), so interpolation always
    extrapolates from real values, preventing error accumulation.
    """
    if skip_this_step and prev_d[-2] is not None:
        if not lagrange_this_step:
            # PSI: linear extrapolation in d space
            d, prev_interp = _zeus_psi_interp_d(prev_d, cons_skip, prev_interp, caching_mode)
            denoised = x - d * utils.append_dims(sigma, x.ndim)
        else:
            # Lagrange interpolation on denoised/x_0
            valid_points = [(s, v) for s, v in zip(lagrange_step_list, lagrange_x0_list)
                           if s is not None and v is not None]
            if len(valid_points) >= 2:
                steps_t = [p[0] for p in valid_points]
                vals = [p[1] for p in valid_points]
                denoised = lagrange_skip(steps_t, vals, i)
            else:
                denoised = model(x, sigma * s_in, **extra_args)
            d = to_d(x, sigma, denoised)
        was_skipped = True
    else:
        denoised = model(x, sigma * s_in, **extra_args)
        d = to_d(x, sigma, denoised)
        was_skipped = False

    return denoised, d, was_skipped, prev_interp


# ===================== ZEUS Sampler Functions =====================

@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                 denominator=3, modular=(0, 1),
                 acc_start=10, acc_end=45,
                 interp_mode="psi", caching_mode="reuse_interp",
                 max_interval=4,
                 lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated Euler sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / N, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigma_hat, s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None,
                s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                denominator=3, modular=(0, 1),
                acc_start=10, acc_end=45,
                interp_mode="psi", caching_mode="reuse_interp",
                max_interval=4,
                lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated Heun sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / N, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigma_hat, s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

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
            x = x + d_prime * dt

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None,
                    denominator=3, modular=(0, 1),
                    acc_start=10, acc_end=45,
                    interp_mode="psi", caching_mode="reuse_interp",
                    max_interval=4,
                    lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DPM++ 2M sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_ddim(model, x, sigmas, extra_args=None, callback=None, disable=None,
                denominator=3, modular=(0, 1),
                acc_start=10, acc_end=45,
                interp_mode="psi", caching_mode="reuse_interp",
                max_interval=4,
                lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DDIM sampler (equivalent to Euler for this formulation)."""
    return sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                        denominator=denominator, modular=modular,
                        acc_start=acc_start, acc_end=acc_end,
                        interp_mode=interp_mode, caching_mode=caching_mode,
                        max_interval=max_interval,
                        lagrange_term=lagrange_term, lagrange_int=lagrange_int,
                        lagrange_step=lagrange_step)


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                 denominator=3, modular=(0, 1),
                 acc_start=10, acc_end=45,
                 interp_mode="psi", caching_mode="reuse_interp",
                 max_interval=4,
                 lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DPM-Solver-2 sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / N, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigma_hat, s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

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
            x = x + d_2 * dt_2

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4,
               denominator=3, modular=(0, 1),
               acc_start=10, acc_end=45,
               interp_mode="psi", caching_mode="reuse_interp",
               max_interval=4,
               lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated Linear Multi-Step sampler."""
    from scipy import integrate
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []

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

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_ipndm(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=4,
                 denominator=3, modular=(0, 1),
                 acc_start=10, acc_end=45,
                 interp_mode="psi", caching_mode="reuse_interp",
                 max_interval=4,
                 lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated iPNDM sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    buffer_model = []

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next

        denoised, d_cur_full, was_skipped, prev_interp = _zeus_handle_skip(
            x_cur, t_cur, s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d_cur_full, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        if callback is not None:
            callback({'x': x_cur, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i + 1)
        if t_next == 0:
            x_next = denoised
        elif order == 1:
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x_next


@torch.no_grad()
def sample_ipndm_v(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=4,
                   denominator=3, modular=(0, 1),
                   acc_start=10, acc_end=45,
                   interp_mode="psi", caching_mode="reuse_interp",
                   max_interval=4,
                   lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated iPNDM-v sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    t_steps = sigmas
    buffer_model = []

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next

        denoised, d_cur_full, was_skipped, prev_interp = _zeus_handle_skip(
            x_cur, t_cur, s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d_cur_full, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        if callback is not None:
            callback({'x': x_cur, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i + 1)
        if t_next == 0:
            x_next = denoised
        elif order == 1:
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i - 1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i - 1])
            h_n_2 = (t_steps[i - 1] - t_steps[i - 2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i - 1])
            h_n_2 = (t_steps[i - 1] - t_steps[i - 2])
            h_n_3 = (t_steps[i - 2] - t_steps[i - 3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x_next


@torch.no_grad()
def sample_deis(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=3, deis_mode='tab',
                denominator=3, modular=(0, 1),
                acc_start=10, acc_end=45,
                interp_mode="psi", caching_mode="reuse_interp",
                max_interval=4,
                lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DEIS sampler."""
    from comfy.k_diffusion import deis
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    t_steps = sigmas
    coeff_list = deis.get_deis_coeff_list(t_steps, max_order, deis_mode=deis_mode)
    buffer_model = []

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next

        denoised, d_cur_full, was_skipped, prev_interp = _zeus_handle_skip(
            x_cur, t_cur, s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d_cur_full, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        if callback is not None:
            callback({'x': x_cur, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i + 1)
        if t_next <= 0:
            order = 1
        if order == 1:
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x_next


def _default_noise_sampler(x, seed=None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None
    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


def _get_ancestral_step(sigma_from, sigma_to, eta=1.):
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None,
                         denominator=3, modular=(0, 1),
                         acc_start=10, acc_end=45,
                         interp_mode="psi", caching_mode="reuse_interp",
                         max_interval=4,
                         lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated RES Multistep sampler (eta=0, no cfg_pp)."""
    import comfy.model_patcher
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = _default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t
    old_sigma_down = None
    old_denoised = None

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        sigma_down, sigma_up = _get_ancestral_step(sigmas[i], sigmas[i + 1], eta=0.)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0 or old_denoised is None:
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        old_denoised = denoised
        old_sigma_down = sigma_down

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_gradient_estimation(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.,
                               denominator=3, modular=(0, 1),
                               acc_start=10, acc_end=45,
                               interp_mode="psi", caching_mode="reuse_interp",
                               max_interval=4,
                               lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated Gradient Estimation sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        dt = sigmas[i + 1] - sigmas[i]
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            x = x + d * dt
            if i >= 1 and old_d is not None:
                d_bar = (ge_gamma - 1) * (d - old_d)
                x = x + d_bar * dt
        old_d = d

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None,
                           denominator=3, modular=(0, 1),
                           acc_start=10, acc_end=45,
                           interp_mode="psi", caching_mode="reuse_interp",
                           max_interval=4,
                           lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated Euler Ancestral sampler."""
    import comfy.model_sampling
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_euler_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler,
                                         denominator=denominator, modular=modular, acc_start=acc_start, acc_end=acc_end,
                                         interp_mode=interp_mode, caching_mode=caching_mode, max_interval=max_interval,
                                         lagrange_term=lagrange_term, lagrange_int=lagrange_int, lagrange_step=lagrange_step)
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = _default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        sigma_down, sigma_up = _get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigma_down == 0:
            x = denoised
        else:
            dt = sigma_down - sigmas[i]
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_euler_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None,
                              denominator=3, modular=(0, 1),
                              acc_start=10, acc_end=45,
                              interp_mode="psi", caching_mode="reuse_interp",
                              max_interval=4,
                              lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated Euler Ancestral RF sampler."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = _default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1]**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2)**0.5
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None,
                           denominator=3, modular=(0, 1),
                           acc_start=10, acc_end=45,
                           interp_mode="psi", caching_mode="reuse_interp",
                           max_interval=4,
                           lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DPM-2 Ancestral sampler."""
    import comfy.model_sampling
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_dpm_2_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler,
                                          denominator=denominator, modular=modular, acc_start=acc_start, acc_end=acc_end,
                                          interp_mode=interp_mode, caching_mode=caching_mode, max_interval=max_interval,
                                          lagrange_term=lagrange_term, lagrange_int=lagrange_int, lagrange_step=lagrange_step)
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = _default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        sigma_down, sigma_up = _get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
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
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_dpm_2_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None,
                              denominator=3, modular=(0, 1),
                              acc_start=10, acc_end=45,
                              interp_mode="psi", caching_mode="reuse_interp",
                              max_interval=4,
                              lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DPM-2 Ancestral RF sampler."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = _default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    N = len(sigmas) - 1
    (acc_start, acc_end, use_lagrange, modular_set,
     prev_d, skip_this_step, cons_skip, prev_interp, lagrange_this_step,
     lagrange_x0_list, lagrange_step_list) = _zeus_init(N, acc_start, acc_end, lagrange_term, modular)

    for i in trange(N, disable=disable):
        denoised, d, was_skipped, prev_interp = _zeus_handle_skip(
            x, sigmas[i], s_in, model, extra_args,
            skip_this_step, lagrange_this_step,
            prev_d, cons_skip, prev_interp, caching_mode,
            lagrange_x0_list, lagrange_step_list, i)

        _zeus_update_d_history(prev_d, d, was_skipped)
        _zeus_update_lagrange(use_lagrange, lagrange_int, i, denoised,
                              lagrange_term, lagrange_x0_list, lagrange_step_list)

        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (sigmas[i + 1]**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2)**0.5

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
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
            x = x + d_2 * dt_2
            x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff

        skip_this_step, cons_skip, lagrange_this_step = _zeus_should_skip(
            i, prev_d, cons_skip,
            denominator, modular_set, acc_start, acc_end,
            max_interval,
            use_lagrange, lagrange_int, lagrange_step)

    return x


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None,
                    denominator=3, modular=(0, 1),
                    acc_start=10, acc_end=45,
                    interp_mode="psi", caching_mode="reuse_interp",
                    max_interval=4,
                    lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DPM-Fast sampler (wraps euler with sigma schedule)."""
    sigmas = torch.linspace(sigma_max.log() if isinstance(sigma_max, torch.Tensor) else torch.tensor(sigma_max).log(),
                            sigma_min.log() if isinstance(sigma_min, torch.Tensor) else torch.tensor(sigma_min).log(),
                            n + 1, device=x.device).exp()
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                        denominator=denominator, modular=modular,
                        acc_start=acc_start, acc_end=acc_end,
                        interp_mode=interp_mode, caching_mode=caching_mode,
                        max_interval=max_interval,
                        lagrange_term=lagrange_term, lagrange_int=lagrange_int,
                        lagrange_step=lagrange_step)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None,
                        denominator=3, modular=(0, 1),
                        acc_start=10, acc_end=45,
                        interp_mode="psi", caching_mode="reuse_interp",
                        max_interval=4,
                        lagrange_term=0, lagrange_int=None, lagrange_step=None):
    """ZEUS-accelerated DPM-Adaptive sampler (wraps euler with sigma schedule)."""
    sigmas = torch.linspace(sigma_max.log() if isinstance(sigma_max, torch.Tensor) else torch.tensor(sigma_max).log(),
                            sigma_min.log() if isinstance(sigma_min, torch.Tensor) else torch.tensor(sigma_min).log(),
                            n + 1, device=x.device).exp()
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                        denominator=denominator, modular=modular,
                        acc_start=acc_start, acc_end=acc_end,
                        interp_mode=interp_mode, caching_mode=caching_mode,
                        max_interval=max_interval,
                        lagrange_term=lagrange_term, lagrange_int=lagrange_int,
                        lagrange_step=lagrange_step)


# Aliases
sample_heunpp2 = sample_heun
