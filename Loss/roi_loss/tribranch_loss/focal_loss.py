import torch
from torch import nn
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        reduction: str = 'none',
        *,
        combine: str = 'add',
        lambda_vp: float = 0.5,
        lambda_den: float = 0.5,
        vp_sigma: float = 0.5,
        den_gamma: float = 1.0,
        img_size: Optional[Tuple[int, int]] = None,
        use_vp_weight: bool = False,
        use_den_weight: bool = False,
        normalize_geo_weight: bool = True,
        weight_min: Optional[float] = 0.5,
        weight_max: Optional[float] = 2.0,
        debug_gadw: bool = False,
        strict_gadw: bool = False,
    ):
        super().__init__()
        self.register_buffer(name='alpha', tensor=alpha)

        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6

        self.combine = combine
        assert self.combine in ('add', 'mul', 'sum')

        self.lambda_vp = float(lambda_vp)
        self.lambda_den = float(lambda_den)
        self.vp_sigma = float(vp_sigma)
        self.den_gamma = float(den_gamma)

        self.use_vp_weight = use_vp_weight
        self.use_den_weight = use_den_weight
        self.normalize_geo_weight = normalize_geo_weight
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.debug_gadw = debug_gadw
        self.strict_gadw = strict_gadw

        if img_size is not None:
            assert len(img_size) == 2
        self.img_size = img_size

    def _compute_vp_weight(self, vp_dist=None, anchor_xy=None, vp_xy=None, max_dist=None):
        if vp_dist is None:
            if anchor_xy is None or vp_xy is None:
                return None
            vp_dist = torch.linalg.norm(anchor_xy - vp_xy, dim=-1)

        if max_dist is None:
            if self.img_size is not None:
                img_h, img_w = self.img_size
                diag = (img_h ** 2 + img_w ** 2) ** 0.5
                max_dist = vp_dist.new_tensor(diag)
            else:
                d = vp_dist.clamp_min(self.eps)
                d_norm = d / (d + 1.0)
                return torch.exp(-(d_norm ** 2) / (2.0 * self.vp_sigma ** 2 + self.eps))

        d_norm = vp_dist / (max_dist + self.eps)
        return torch.exp(-(d_norm ** 2) / (2.0 * self.vp_sigma ** 2 + self.eps))

    def _compute_density_weight(self, lane_density):
        if lane_density is None:
            return None

        d_min = torch.amin(lane_density)
        d_max = torch.amax(lane_density)

        if torch.abs(d_max - d_min) < self.eps:
            return torch.zeros_like(lane_density)

        den_norm = (lane_density - d_min) / (d_max - d_min + self.eps)
        return den_norm.clamp(0.0, 1.0) ** self.den_gamma

    def _expand_weight(self, w, shape_like):
        if w.shape == shape_like.shape:
            return w
        return w.expand_as(shape_like)

    def _stabilize_geo_weight(self, w):
        if self.normalize_geo_weight:
            w = w / (w.mean().detach() + self.eps)

        if self.weight_min is not None or self.weight_max is not None:
            min_val = self.weight_min if self.weight_min is not None else -float('inf')
            max_val = self.weight_max if self.weight_max is not None else float('inf')
            w = torch.clamp(w, min=min_val, max=max_val)

        return w

    def _combine_geo_weights(self, w_vp, w_den, shape_like):
        weights = []
        lambdas = []

        if w_vp is not None:
            weights.append(w_vp)
            lambdas.append(self.lambda_vp)

        if w_den is not None:
            weights.append(w_den)
            lambdas.append(self.lambda_den)

        if len(weights) == 0:
            return torch.ones_like(shape_like)

        if self.combine == 'add':
            w = torch.ones_like(shape_like)
            for wi, li in zip(weights, lambdas):
                if li > 0:
                    w = w + li * self._expand_weight(wi, shape_like)

        elif self.combine == 'mul':
            w = torch.ones_like(shape_like)
            for wi, li in zip(weights, lambdas):
                if li > 0:
                    wi = self._expand_weight(wi, shape_like)
                    w = w * (wi.clamp_min(self.eps) ** li)

        elif self.combine == 'sum':
            denom = sum([li for li in lambdas if li > 0])
            if denom <= 0:
                return torch.ones_like(shape_like)

            w = torch.zeros_like(shape_like)
            for wi, li in zip(weights, lambdas):
                if li > 0:
                    w = w + li * self._expand_weight(wi, shape_like)
            w = w / (denom + self.eps)

        else:
            raise ValueError(f"Invalid combine mode: {self.combine}")

        return self._stabilize_geo_weight(w)

    def forward(
        self,
        pred,
        target,
        *,
        geo_weight=None,
        vp_dist=None,
        lane_density=None,
        anchor_xy=None,
        vp_xy=None,
        max_dist=None,
    ):
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        target = target.float()

        focal_pos = -self.alpha[1] * torch.pow(1.0 - pred, self.gamma) * torch.log(pred)
        focal_neg = -self.alpha[0] * torch.pow(pred, self.gamma) * torch.log(1.0 - pred)
        loss = target * focal_pos + (1.0 - target) * focal_neg

        if geo_weight is not None:
            w = self._expand_weight(geo_weight, loss)
        else:
            w_vp = None
            w_den = None

            if self.use_vp_weight:
                w_vp = self._compute_vp_weight(vp_dist, anchor_xy, vp_xy, max_dist)
                if self.strict_gadw and w_vp is None:
                    raise RuntimeError(
                        "GADW error: use_vp_weight=True, but vp_dist or anchor_xy+vp_xy was not provided."
                    )

            if self.use_den_weight:
                w_den = self._compute_density_weight(lane_density)
                if self.strict_gadw and w_den is None:
                    raise RuntimeError(
                        "GADW error: use_den_weight=True, but lane_density was not provided."
                    )

            w = self._combine_geo_weights(w_vp, w_den, loss)


        loss = loss * w

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
