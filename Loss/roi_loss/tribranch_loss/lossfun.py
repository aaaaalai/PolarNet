import math
import torch
from torch import nn
import torch.nn.functional as F

from .focal_loss import FocalLoss
from .rank_loss import RankLoss
from .lineiou_loss import liou_loss
from .assign import Assigner
from utils.coord_transform import CoordTrans_torch


class TriBranchLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cls_loss_weight = cfg.cls_loss_weight
        self.iou_loss_weight = cfg.iou_loss_weight
        self.end_loss_weight = cfg.end_loss_weight
        self.aux_loss_weight = cfg.aux_loss_weight
        self.rank_loss_weight = cfg.rank_loss_weight

        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.center_w, self.center_h = cfg.center_w, cfg.center_h
        self.num_offsets = cfg.num_offsets
        self.n_strips = self.num_offsets - 1
        self.offset_stride = cfg.offset_stride
        self.conf_thres = cfg.conf_thres
        self.conf_thres_o2o = cfg.conf_thres_o2o
        self.loss_iou_width = cfg.loss_iou_width
        self.g_weight = cfg.g_weight

        self.gadw_combine = getattr(cfg, 'gadw_combine', 'add')
        self.lambda_vp = getattr(cfg, 'lambda_vp', 0.5)
        self.lambda_den = getattr(cfg, 'lambda_den', 0.5)
        self.vp_sigma = getattr(cfg, 'vp_sigma', 0.5)
        self.den_gamma = getattr(cfg, 'den_gamma', 1.0)
        self.use_vp_weight = getattr(cfg, 'use_vp_weight', False)
        self.use_den_weight = getattr(cfg, 'use_den_weight', False)
        self.debug_gadw = getattr(cfg, 'debug_gadw', False)
        self.strict_gadw = getattr(cfg, 'strict_gadw', False)
        self.density_radius = getattr(cfg, 'density_radius', 30.0)

        alpha = torch.tensor([1 - cfg.cls_loss_alpha, cfg.cls_loss_alpha])
        alpha_o2o = torch.tensor([1 - cfg.cls_loss_alpha_o2o, cfg.cls_loss_alpha_o2o])

        self.coord_trans = CoordTrans_torch(cfg)
        self.assigner = Assigner(cfg)

        self.cls_criterion = FocalLoss(
            alpha=alpha,
            gamma=2.,
            combine=self.gadw_combine,
            lambda_vp=self.lambda_vp,
            lambda_den=self.lambda_den,
            vp_sigma=self.vp_sigma,
            den_gamma=self.den_gamma,
            img_size=(self.img_h, self.img_w),
            use_vp_weight=self.use_vp_weight,
            use_den_weight=self.use_den_weight,
            debug_gadw=self.debug_gadw,
            strict_gadw=self.strict_gadw,
        )

        self.cls_criterion_o2o = FocalLoss(
            alpha=alpha_o2o,
            gamma=2.,
            combine=self.gadw_combine,
            lambda_vp=self.lambda_vp,
            lambda_den=self.lambda_den,
            vp_sigma=self.vp_sigma,
            den_gamma=self.den_gamma,
            img_size=(self.img_h, self.img_w),
            use_vp_weight=self.use_vp_weight,
            use_den_weight=self.use_den_weight,
            debug_gadw=self.debug_gadw,
            strict_gadw=self.strict_gadw,
        )

        self.rank_loss = RankLoss(tau=0.5)
        self.y_stride = (
            self.offset_stride
            * ((cfg.ori_img_h - cfg.cut_height) / cfg.ori_img_w)
            / (self.img_h / self.img_w)
        )

    def compute_gadw_inputs(
        self,
        lane_point_xs_gt,
        lane_point_validmask,
        lanereg_base_car,
    ):
        device = lanereg_base_car.device
        num_priors = lanereg_base_car.shape[0]

        vp_dist = lanereg_base_car.new_zeros(num_priors)
        lane_density = lanereg_base_car.new_zeros(num_priors)

        if lane_point_xs_gt.numel() == 0:
            return vp_dist, lane_density

        if lanereg_base_car.dim() == 3:
            anchor_xs = lanereg_base_car[..., 0]
        else:
            anchor_xs = lanereg_base_car

        num_offsets = lane_point_xs_gt.shape[-1]
        y_coords = torch.arange(
            num_offsets,
            device=device,
            dtype=lane_point_xs_gt.dtype
        ) * self.y_stride

        line_params = []

        for lane_xs, valid_mask in zip(lane_point_xs_gt, lane_point_validmask):
            valid_mask = valid_mask.bool()

            if valid_mask.sum() < 2:
                continue

            xs = lane_xs[valid_mask]
            ys = y_coords[valid_mask]

            y_mean = ys.mean()
            x_mean = xs.mean()

            dy = ys - y_mean
            dx = xs - x_mean

            denom = (dy * dy).sum().clamp_min(1e-6)
            a = (dy * dx).sum() / denom
            b = x_mean - a * y_mean

            line_params.append((a, b))

        if len(line_params) == 0:
            return vp_dist, lane_density

        anchor_valid = torch.isfinite(anchor_xs)
        anchor_xs_safe = torch.where(anchor_valid, anchor_xs, torch.zeros_like(anchor_xs))

        anchor_valid_count = anchor_valid.float().sum(dim=-1).clamp_min(1.0)
        anchor_center_x = (anchor_xs_safe * anchor_valid.float()).sum(dim=-1) / anchor_valid_count

        y_expand = y_coords.unsqueeze(0).expand_as(anchor_xs_safe)
        anchor_center_y = (y_expand * anchor_valid.float()).sum(dim=-1) / anchor_valid_count

        if len(line_params) >= 2:
            vp_candidates = []

            for i in range(len(line_params)):
                a1, b1 = line_params[i]

                for j in range(i + 1, len(line_params)):
                    a2, b2 = line_params[j]

                    denom = a1 - a2
                    if torch.abs(denom) < 1e-6:
                        continue

                    y_vp = (b2 - b1) / denom
                    x_vp = a1 * y_vp + b1

                    if torch.isfinite(x_vp) and torch.isfinite(y_vp):
                        vp_candidates.append(torch.stack([x_vp, y_vp]))

            if len(vp_candidates) > 0:
                vp_xy = torch.stack(vp_candidates, dim=0).mean(dim=0)
            else:
                valid_points = lane_point_validmask.bool()
                vp_xy = torch.stack([
                    lane_point_xs_gt[valid_points].mean(),
                    y_coords.new_tensor(0.0)
                ])
        else:
            a, b = line_params[0]
            y_vp = y_coords.new_tensor(0.0)
            x_vp = a * y_vp + b
            vp_xy = torch.stack([x_vp, y_vp])

        anchor_xy = torch.stack([anchor_center_x, anchor_center_y], dim=-1)
        vp_dist = torch.linalg.norm(anchor_xy - vp_xy.unsqueeze(0), dim=-1)

        density_values = []

        for lane_xs, valid_mask in zip(lane_point_xs_gt, lane_point_validmask):
            valid_mask = valid_mask.bool()

            if valid_mask.sum() == 0:
                continue

            common_mask = valid_mask.unsqueeze(0) & anchor_valid

            diff = torch.abs(anchor_xs_safe - lane_xs.unsqueeze(0))
            diff = torch.where(common_mask, diff, torch.full_like(diff, 1e6))

            min_diff = diff.min(dim=-1)[0]
            density_values.append(torch.exp(-min_diff / self.density_radius))

        if len(density_values) > 0:
            lane_density = torch.stack(density_values, dim=0).sum(dim=0)

        return vp_dist.detach(), lane_density.detach()

    def forward(self, pred_dict, target_dict):
        cls_pred_batch = pred_dict['cls']
        end_points_batch = pred_dict['end_points']
        lanereg_xs_offset_batch = pred_dict['lanereg_xs_offset']
        cls_pred_batch_o2o = pred_dict['cls_o2o']
        line_paras_group_reg_batch = pred_dict['line_paras_group_reg']
        lanereg_base_car_batch = pred_dict['lanereg_base_car']
        anchor_embeddings_batch = pred_dict['anchor_embeddings']

        line_paras_group_gt_batch = target_dict['line_paras_group']
        group_validmask_batch = target_dict['group_validmask']
        lane_valids_batch = target_dict['lane_valid']
        lane_point_xs_gt_batch = target_dict['lane_point_xs']
        lane_point_validmask_batch = target_dict['lane_point_validmask']
        end_point_gt_batch = target_dict['end_point']
        line_paras_batch = target_dict['line_paras']

        cls_loss = torch.tensor([0.], device=cls_pred_batch.device)
        cls_o2o_loss = torch.tensor([0.], device=cls_pred_batch.device)
        iou_loss = torch.tensor([0.], device=cls_pred_batch.device)
        end_point_loss = torch.tensor([0.], device=cls_pred_batch.device)
        aux_reg_loss = torch.tensor([0.], device=cls_pred_batch.device)
        rank_loss = torch.tensor([0.], device=cls_pred_batch.device)

        batch_size = cls_pred_batch.shape[0]

        prior_idx_list = []
        gt_idx_list = []
        batch_idx_list = []

        batch_idx = 0

        for (
            cls_pred,
            cls_pred_o2o,
            lanereg_xs_offset,
            lanereg_base_car,
            lane_valids,
            lane_point_xs_gt,
            lane_point_validmask,
            anchor_embeddings,
            line_paras,
        ) in zip(
            cls_pred_batch,
            cls_pred_batch_o2o,
            lanereg_xs_offset_batch,
            lanereg_base_car_batch,
            lane_valids_batch,
            lane_point_xs_gt_batch,
            lane_point_validmask_batch,
            anchor_embeddings_batch,
            line_paras_batch,
        ):

            num_gt = lane_valids.sum()

            lane_point_xs_gt = lane_point_xs_gt[lane_valids]
            lane_point_validmask = lane_point_validmask[lane_valids]
            line_paras = line_paras[lane_valids]

            cls_target = cls_pred.new_zeros(cls_pred.shape[0]).long()
            cls_target_o2o = cls_pred.new_zeros(cls_pred.shape[0]).long()

            if num_gt == 0:
                geo_weight = cls_pred.new_ones(cls_pred.shape)
                geo_weight_o2o = cls_pred_o2o.new_ones(cls_pred_o2o.shape)

                cls_loss = cls_loss + self.cls_criterion(
                    cls_pred,
                    cls_target,
                    geo_weight=geo_weight
                ).sum()

                cls_mask_o2o = cls_pred > self.conf_thres_o2o

                cls_o2o_loss = cls_o2o_loss + (
                    self.cls_criterion_o2o(
                        cls_pred_o2o,
                        cls_target_o2o,
                        geo_weight=geo_weight_o2o
                    ) * cls_mask_o2o
                ).sum()

                batch_idx += 1
                continue

            with torch.no_grad():
                prior_idx, prior_idx_o2o, prior_idx_reg, gt_idx_reg = self.assigner(
                    cls_pred,
                    cls_pred_o2o,
                    lanereg_xs_offset,
                    lanereg_base_car,
                    lane_point_xs_gt,
                    lane_point_validmask,
                    anchor_embeddings,
                    line_paras,
                )

                if prior_idx_reg.numel() > 0:
                    prior_idx_list.append(prior_idx_reg)
                    gt_idx_list.append(gt_idx_reg)
                    batch_idx_list.append(batch_idx * torch.ones_like(prior_idx_reg))

                cls_target[prior_idx] = 1
                cls_target_o2o[prior_idx_o2o] = 1

            vp_dist, lane_density = self.compute_gadw_inputs(
                lane_point_xs_gt=lane_point_xs_gt,
                lane_point_validmask=lane_point_validmask,
                lanereg_base_car=lanereg_base_car,
            )

            if self.debug_gadw and (self.use_vp_weight or self.use_den_weight):
                with torch.no_grad():
                    print(
                        '[GADW input check] '
                        f'vp_dist min={vp_dist.min().item():.4f}, '
                        f'vp_dist max={vp_dist.max().item():.4f}, '
                        f'vp_dist std={vp_dist.std().item():.4f}, '
                        f'density min={lane_density.min().item():.4f}, '
                        f'density max={lane_density.max().item():.4f}, '
                        f'density std={lane_density.std().item():.4f}'
                    )

            cls_loss = cls_loss + self.cls_criterion(
                cls_pred,
                cls_target,
                vp_dist=vp_dist,
                lane_density=lane_density,
            ).sum()

            cls_mask_o2o = cls_pred > self.conf_thres_o2o

            cls_o2o_loss = cls_o2o_loss + (
                self.cls_criterion_o2o(
                    cls_pred_o2o,
                    cls_target_o2o,
                    vp_dist=vp_dist,
                    lane_density=lane_density,
                ) * cls_mask_o2o
            ).sum()

            rank_loss = rank_loss + self.rank_loss(
                cls_pred_o2o,
                cls_target_o2o,
                mask=cls_mask_o2o
            )

            batch_idx += 1

        if len(prior_idx_list) > 0:
            prior_idx_batch = (
                torch.cat(batch_idx_list, dim=0),
                torch.cat(prior_idx_list, dim=0)
            )

            gt_idx_batch = (
                torch.cat(batch_idx_list, dim=0),
                torch.cat(gt_idx_list, dim=0)
            )

            with torch.no_grad():
                line_paras_group_gt = line_paras_group_gt_batch[gt_idx_batch].detach().clone()
                group_validmask = group_validmask_batch[gt_idx_batch].detach().clone().unsqueeze(-1)

                line_paras_group_gt = line_paras_group_gt * group_validmask

                end_points_gt = end_point_gt_batch[gt_idx_batch]
                x_samples_car = lanereg_base_car_batch[prior_idx_batch][..., 0].detach().clone()

                lane_points_target = lane_point_xs_gt_batch[gt_idx_batch]
                lane_points_validmask = lane_point_validmask_batch[gt_idx_batch].bool()

                end_points_gt = end_points_gt / self.img_h * self.n_strips
                line_paras_group_gt[..., 0] *= 180 / math.pi

            end_points = end_points_batch[prior_idx_batch]
            lanereg_xs_offset = lanereg_xs_offset_batch[prior_idx_batch]
            line_paras_group_reg = line_paras_group_reg_batch[prior_idx_batch]

            line_paras_group_reg = line_paras_group_reg * group_validmask
            line_paras_group_reg[..., 0] *= 180
            line_paras_group_reg[..., 1] *= self.img_w

            iou_loss = iou_loss + liou_loss(
                lanereg_xs_offset * self.img_w + x_samples_car,
                lane_points_target,
                lane_points_validmask,
                width=self.loss_iou_width,
                y_stride=self.y_stride,
                g_weight=self.g_weight,
            ).mean()

            end_point_loss = end_point_loss + F.smooth_l1_loss(
                end_points * self.n_strips,
                end_points_gt
            ).mean()

            aux_reg_loss = aux_reg_loss + F.smooth_l1_loss(
                line_paras_group_reg.flatten(-2, -1),
                line_paras_group_gt.flatten(-2, -1)
            ).mean()

        cls_loss /= batch_size
        cls_o2o_loss /= batch_size

        loss = (
            cls_loss * self.cls_loss_weight
            + iou_loss * self.iou_loss_weight
            + end_point_loss * self.end_loss_weight
            + aux_reg_loss * self.aux_loss_weight
            + cls_o2o_loss * self.cls_loss_weight
            + rank_loss * self.rank_loss_weight
        )

        loss_msg = {
            'loss': loss,
            'cls_loss': cls_loss * self.cls_loss_weight,
            'reg_loss': end_point_loss * self.end_loss_weight + aux_reg_loss * self.aux_loss_weight,
            'iou_loss': iou_loss * self.iou_loss_weight,
            'cls_loss_o2o': cls_o2o_loss * self.cls_loss_weight,
            'rank_loss': rank_loss * self.rank_loss_weight,
        }

        return loss, loss_msg
