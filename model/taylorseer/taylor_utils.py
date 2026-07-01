"""
TaylorSeer / HiCache core utilities: derivative approximation, prediction formula, cache init.
Based on:
- TaylorSeer: https://github.com/Shenyi-Z/TaylorSeer (ICCV 2025)
- HiCache: https://arxiv.org/abs/2508.16984 (ICLR 2026)
"""
from typing import Dict
import torch
import math


def _hermite_polynomial(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Physicist's Hermite polynomial H_n(x).
    Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2 * x

    H_prev = torch.ones_like(x)
    H_curr = 2 * x

    for k in range(2, n + 1):
        H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
        H_prev, H_curr = H_curr, H_next

    return H_curr


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation using finite differences.
    Stores 0th to max_order-th order derivatives in cache.
    """
    if len(current['activated_steps']) < 2:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {0: feature}
        return

    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        prev = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None)
        if prev is not None and current['step'] > cache_dic['first_enhance'] - 2:
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - prev) / difference_distance
        else:
            break

    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Feature prediction dispatcher: chooses between Taylor or HiCache prediction.
    """
    prediction_mode = cache_dic.get('prediction_mode', 'taylor')

    if prediction_mode == 'taylor':
        return _taylor_expansion(cache_dic, current)
    elif prediction_mode == 'hicache':
        return _hicache_prediction(cache_dic, current)
    elif prediction_mode == 'taylor_scaled':
        return _taylor_scaled_prediction(cache_dic, current)
    else:
        raise ValueError(f"Unknown prediction_mode: '{prediction_mode}'")


def _taylor_expansion(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Standard Taylor series expansion prediction.
    F_pred = F_0 + Σ (1/k!) * x^k * Δ^kF
    """
    x = current['step'] - current['activated_steps'][-1]
    output = 0

    cached = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    max_order = cache_dic.get('max_order', 3)
    effective_order = min(max_order + 1, len(cached))

    for i in range(effective_order):
        output += (1 / math.factorial(i)) * cached[i] * (x ** i)

    return output


def _hicache_prediction(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Hermite polynomial-based feature prediction (HiCache).
    F_pred = F_0 + Σ (1/k!) * H_k(σx) * σ^k * Δ^kF
    """
    x = current['step'] - current['activated_steps'][-1]

    cached = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    max_order = cache_dic.get('max_order', 3)
    available_order = len(cached) - 1
    order = min(max_order, available_order)

    if order < 1:
        return cached.get(0)

    F_latest = cached[0].clone()
    x_tensor = torch.tensor(float(x), dtype=F_latest.dtype, device=F_latest.device)

    scale_factor = cache_dic.get('hicache_scale_factor', 0.5)
    x_scaled = x_tensor * scale_factor

    pred = F_latest.clone()

    for k in range(1, order + 1):
        diff_k = cached[k]
        Hk = _hermite_polynomial(x_scaled, k)
        alpha = float(Hk / math.factorial(k)) * (scale_factor ** k)
        pred = pred + diff_k * alpha

    return pred


def _taylor_scaled_prediction(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Taylor prediction with dual-scaling (HiCache's scaling trick applied to Taylor basis).
    F_pred = F_0 + Σ (1/k!) * s^(2k) * x^k * Δ^kF
    """
    x = current['step'] - current['activated_steps'][-1]

    cached = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    max_order = cache_dic.get('max_order', 3)
    available_order = len(cached) - 1
    order = min(max_order, available_order)

    if order < 1:
        return cached.get(0)

    F_latest = cached[0].clone()
    scale = cache_dic.get('hicache_scale_factor', 0.5)

    pred = F_latest.clone()
    for k in range(1, order + 1):
        diff_k = cached[k]
        alpha = (float(x ** k) / math.factorial(k)) * (scale ** (2 * k))
        pred = pred + diff_k * alpha

    return pred


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache storage for derivatives at step 0.
    """
    if current['step'] == 0 and cache_dic['taylor_cache']:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


def force_scheduler(cache_dic: Dict, current: Dict):
    """
    Dynamic threshold scheduler: adjusts cal_threshold based on current step progress.
    """
    step_factor = torch.tensor(1.0)
    threshold = torch.round(torch.tensor(float(cache_dic['fresh_threshold'])) / step_factor)
    cache_dic['cal_threshold'] = threshold


def cal_type(cache_dic: Dict, current: Dict):
    """
    Determine calculation type for this step:
    - 'full': compute all blocks normally, update cache
    - 'Taylor': use Taylor/HiCache expansion to predict output
    """
    first_step = (current['step'] < cache_dic['first_enhance'])

    if not first_step:
        fresh_interval = cache_dic['cal_threshold']
    else:
        fresh_interval = cache_dic['fresh_threshold']

    if first_step or (cache_dic['cache_counter'] == fresh_interval - 1):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        force_scheduler(cache_dic, current)
    elif cache_dic['taylor_cache']:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Taylor'
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'full'


def cache_init(num_double_layers: int, num_single_layers: int, num_steps: int,
               max_order: int = 1, fresh_threshold: int = 6, first_enhance: int = 3,
               prediction_mode: str = 'taylor', hicache_scale_factor: float = 0.5):
    """
    Initialize cache structure for TaylorSeer / HiCache.

    Args:
        num_double_layers: number of double stream (joint attention) blocks
        num_single_layers: number of single stream blocks
        num_steps: total number of inference steps
        max_order: Taylor expansion max order (higher = more accurate but more memory)
        fresh_threshold: interval between full computation steps
        first_enhance: number of initial steps that always do full computation
        prediction_mode: 'taylor', 'hicache', or 'taylor_scaled'
        hicache_scale_factor: scale factor σ for Hermite polynomial (HiCache mode)
    """
    cache_dic = {}
    cache = {}
    cache[-1] = {}
    cache[-1]['double_stream'] = {}
    cache[-1]['single_stream'] = {}
    cache_dic['cache_counter'] = 0

    for j in range(num_double_layers):
        cache[-1]['double_stream'][j] = {}

    for j in range(num_single_layers):
        cache[-1]['single_stream'][j] = {}

    cache_dic['cache'] = cache
    cache_dic['taylor_cache'] = True
    cache_dic['fresh_threshold'] = fresh_threshold
    cache_dic['max_order'] = max_order
    cache_dic['first_enhance'] = first_enhance
    cache_dic['prediction_mode'] = prediction_mode
    cache_dic['hicache_scale_factor'] = hicache_scale_factor

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current
