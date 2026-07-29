from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class ResidualPredictor:
    """
    TC-Pade residual predictor ported from the official TC_Pade implementation.

    It predicts the whole transformer-stack residual, then reconstructs the
    current output as current_input + predicted_residual.
    """

    def __init__(self, order=3, history_size=6, n_threshold=1.4, cache_device="cuda"):
        self.threshold = n_threshold
        self.order = order
        self.history_size = history_size
        self.cache_device = cache_device
        self.residual_history = deque(maxlen=history_size)
        self.current_order = order
        self.stability_factor = 1.0
        self.step_count = 0
        self.total_timesteps = 20
        self.initialized = False
        self.last_input = None

    def reset(self):
        self.residual_history.clear()
        self.current_order = self.order
        self.stability_factor = 1.0
        self.step_count = 0
        self.initialized = False
        self.last_input = None

    def _cache_device_for(self, tensor):
        if self.cache_device == "cpu":
            return torch.device("cpu")
        return tensor.device

    def cosine_similarity(self, a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)
        if torch.norm(a) == 0 or torch.norm(b) == 0:
            return 0.0
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def _adaptive_order_selection(self):
        if len(self.residual_history) < 3:
            return min(2, self.order)

        similarities = []
        for i in range(1, min(3, len(self.residual_history))):
            sim = self.cosine_similarity(self.residual_history[-i], self.residual_history[-i - 1])
            similarities.append(sim)

        avg_similarity = np.mean(similarities)
        if avg_similarity > 0.98:
            return min(self.order, 4)
        if avg_similarity > 0.90:
            return min(self.order, 3)
        return 2

    def _weighted_extrapolation(self):
        weights = [0.4, 0.3, 0.2, 0.1]
        k = min(len(self.residual_history), self.current_order)
        if k == 0:
            return None

        prediction = torch.zeros_like(self.residual_history[-1])
        total_weight = sum(weights[:k])
        for i in range(k):
            prediction += (weights[i] / total_weight) * self.residual_history[-1 - i]
        return prediction * self.stability_factor

    def _pade_prediction(self):
        if len(self.residual_history) < self.current_order + 1:
            return self._weighted_extrapolation()

        try:
            delta = torch.abs(self.residual_history[-1] - self.residual_history[-2])
            avg_magnitude = 0.5 * (torch.abs(self.residual_history[-1]) + torch.abs(self.residual_history[-2]))
            stability_factor = torch.exp(-5.0 * delta / (avg_magnitude + 1e-7))

            b0 = 2 * stability_factor
            b1 = 1 * stability_factor
            curvature_sign = self.curvetest()
            if curvature_sign < 0:
                a1 = 0.1 * stability_factor
            else:
                a1 = -0.1 * stability_factor

            numerator = b0 * self.residual_history[-1] - b1 * self.residual_history[-2]
            denominator = 1.0 + a1 * curvature_sign
            abs_denom = torch.abs(denominator)
            safe_denom = torch.where(
                abs_denom < 1e-5,
                1.0 + a1 * self.residual_history[-1] if len(self.residual_history) >= 2 else 1.0,
                denominator,
            )

            denom_sign = torch.sign(safe_denom)
            safe_denom = denom_sign * torch.maximum(1e-5 * torch.ones_like(abs_denom), abs_denom)
            result = numerator / safe_denom

            recent_avg = 0.6 * self.residual_history[-1] + 0.4 * self.residual_history[-2]
            max_deviation = 0.5 * torch.abs(self.residual_history[-1] - self.residual_history[-2])
            max_deviation = torch.maximum(max_deviation, 0.1 * avg_magnitude)

            deviation = torch.clamp(result - recent_avg, -max_deviation, max_deviation)
            result = recent_avg + deviation

            blend_factor = 0.7 * stability_factor + 0.3
            result = blend_factor * result + (1 - blend_factor) * recent_avg

            if torch.isnan(result).any() or torch.isinf(result).any():
                result = self.residual_history[-1] + 0.5 * (self.residual_history[-1] - self.residual_history[-2])
                if torch.isnan(result).any() or torch.isinf(result).any():
                    return self._weighted_extrapolation()
            return result
        except Exception as exc:
            print(f"[TC-Pade] prediction failed: {exc}")
            return self._weighted_extrapolation()

    def update_history(self, hidden_in, hidden_out):
        cache_device = self._cache_device_for(hidden_in)
        residual = (hidden_out - hidden_in).detach().clone().to(cache_device)
        self.residual_history.append(residual)
        self.last_input = hidden_in.detach().clone().to(cache_device)
        self.step_count += 1

        if len(self.residual_history) >= 3:
            self.current_order = self._adaptive_order_selection()
            last_change = torch.norm(self.residual_history[-1] - self.residual_history[-2]).item()
            self.stability_factor = max(0.8, min(1.2, 1.0 / (1.0 + 10 * last_change)))

        if not self.initialized and len(self.residual_history) >= min(3, self.order):
            self.initialized = True

    def _high_noise_prediction(self):
        return self.residual_history[-1] * 0.9 + self.residual_history[-2] * 0.1

    def _detail_enhanced_prediction(self):
        grad = self.residual_history[-1] - self.residual_history[-2]
        pade_pred = self._pade_prediction()
        return pade_pred + 0.3 * grad

    def predict_residual(self, timestep):
        if not self.initialized or len(self.residual_history) == 0:
            return None

        if timestep > 0.8 * self.total_timesteps:
            return self._detail_enhanced_prediction()
        if timestep > 0.3 * self.total_timesteps:
            return self._pade_prediction()
        return self._high_noise_prediction()

    def predict_output(self, current_input, cur_step):
        residual_pred = self.predict_residual(cur_step)
        if residual_pred is None or self.last_input is None:
            return None

        residual_pred = residual_pred.to(current_input.device, dtype=current_input.dtype)
        last_input = self.last_input.to(current_input.device, dtype=current_input.dtype)
        input_delta = current_input - last_input
        adjusted_residual = residual_pred + 0.2 * input_delta
        return current_input + adjusted_residual

    def curvetest(self):
        if len(self.residual_history) < 5:
            return True

        def bend(r0, r1, r2, eps=1e-12):
            v1 = (r1 - r0).flatten()
            v2 = (r2 - r1).flatten()
            n1 = torch.norm(v1) + eps
            n2 = torch.norm(v2) + eps
            u1, u2 = v1 / n1, v2 / n2
            return 0.5 * torch.norm(u1 - u2)

        score = bend(self.residual_history[-3], self.residual_history[-2], self.residual_history[-1])
        return bool(score < self.threshold)
