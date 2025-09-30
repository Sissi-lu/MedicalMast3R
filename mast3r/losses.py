# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of MASt3R training losses
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.losses import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from dust3r.losses import Regr3D as Regr3D_dust3r
from dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud)
from dust3r.inference import get_pred_pts3d
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
import lietorch
import math

def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz

def skew_sym(x):
    b = x.shape[:-1]
    x, y, z = x.unbind(dim=-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).view(*b, 3, 3)

class Regr3D (Regr3D_dust3r):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, opt_fit_gt=False,
                 sky_loss_value=2, max_metric_scale=False, loss_in_log=False):
        self.loss_in_log = loss_in_log
        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        super().__init__(criterion, self.norm_mode, gt_scale)

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        if self.loss_in_log == 'before':
            # this only make sense when depth_mode == 'linear'
            gt_pts1 = apply_log_to_norm(gt_pts1)
            gt_pts2 = apply_log_to_norm(gt_pts2)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()

        if not self.norm_all:
            if self.max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(valid1, torch.linalg.norm(gt_pts1, dim=-1), 0).view(B, -1)
                dist2_to_cam1 = torch.where(valid2, torch.linalg.norm(gt_pts2, dim=-1), 0).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1['is_metric_scale'] = gt1['is_metric_scale'] \
                    & (dist1_to_cam1.max(dim=-1).values < self.max_metric_scale) \
                    & (dist2_to_cam1.max(dim=-1).values < self.max_metric_scale)
                gt2['is_metric_scale'] = gt1['is_metric_scale']

            mask = ~gt1['is_metric_scale']
        else:
            mask = torch.ones_like(gt1['is_metric_scale'])
        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(pr_pts1[mask], pr_pts2[mask], self.norm_mode,
                                                                valid1[mask], valid2[mask])

        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2, norm_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode,
                                                                 valid1, valid2, ret_factor=True)
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # add the sky pixel as "valid" pixels...
            mask1 = mask1 | sky1
            mask2 = mask2 | sky2

        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        if self.loss_in_log and self.loss_in_log != 'before':
            # this only make sense when depth_mode == 'exp'
            pred_pts1 = apply_log_to_norm(pred_pts1)
            gt_pts1 = apply_log_to_norm(gt_pts1)
        l1 = self.criterion(pred_pts1, gt_pts1)

        # loss on gt2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        if self.loss_in_log and self.loss_in_log != 'before':
            pred_pts2 = apply_log_to_norm(pred_pts2)
            gt_pts2 = apply_log_to_norm(gt_pts2)
        l2 = self.criterion(pred_pts2, gt_pts2)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # ... but force the loss to be high there
            l1 = torch.where(sky1[mask1], self.sky_loss_value, l1)
            l2 = torch.where(sky2[mask2], self.sky_loss_value, l2)
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth scale.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass


def get_similarities(desc1, desc2, euc=False):
    if euc:  # euclidean distance in same range than similarities
        dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        sim = 1 / (1 + dists)
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim


class MatchingCriterion(BaseCriterion):
    def __init__(self, reduction='mean', fp=torch.float32):
        super().__init__(reduction)
        self.fp = fp

    def forward(self, a, b, valid_matches=None, euc=False):
        assert a.ndim >= 2 and 1 <= a.shape[-1], f'Bad shape = {a.shape}'
        dist = self.loss(a.to(self.fp), b.to(self.fp), valid_matches, euc=euc)
        # one dimension less or reduction to single value
        assert (valid_matches is None and dist.ndim == a.ndim -
                1) or self.reduction in ['mean', 'sum', '1-mean', 'none']
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        if self.reduction == '1-mean':
            return 1. - dist.mean() if dist.numel() > 0 else dist.new_ones(())
        raise ValueError(f'bad {self.reduction=} mode')

    def loss(self, a, b, valid_matches=None):
        raise NotImplementedError


class InfoNCE(MatchingCriterion):
    def __init__(self, temperature=0.07, eps=1e-8, mode='all', **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.eps = eps
        assert mode in ['all', 'proper', 'dual']
        self.mode = mode

    def loss(self, desc1, desc2, valid_matches=None, euc=False):
        # valid positives are along diagonals
        B, N, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2
        if valid_matches is None:
            valid_matches = torch.ones([B, N], dtype=bool)
        # torch.all(valid_matches.sum(dim=-1) > 0) some pairs have no matches????
        assert valid_matches.shape == torch.Size([B, N]) and valid_matches.sum() > 0, "valid matches: %s, torch.size(%d,%d), valid_matches: %d"%(valid_matches.shape, B, N, valid_matches.sum())

        # Tempered similarities
        sim = get_similarities(desc1, desc2, euc) / self.temperature
        sim[sim.isnan()] = -torch.inf  # ignore nans
        # Softmax of positives with temperature
        sim = sim.exp_()  # save peak memory
        positives = sim.diagonal(dim1=-2, dim2=-1)

        # Loss
        if self.mode == 'all':            # Previous InfoNCE
            loss = -torch.log((positives / sim.sum(dim=-1).sum(dim=-1, keepdim=True)).clip(self.eps))
        elif self.mode == 'proper':  # Proper InfoNCE
            loss = -(torch.log((positives / sim.sum(dim=-2)).clip(self.eps)) +
                     torch.log((positives / sim.sum(dim=-1)).clip(self.eps)))
        elif self.mode == 'dual':  # Dual Softmax
            loss = -(torch.log((positives**2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps)))
        else:
            raise ValueError("This should not happen...")
        return loss[valid_matches]


class APLoss (MatchingCriterion):
    """ AP loss
    """

    def __init__(self, nq='torch', min=0, max=1, euc=False, **kw):
        super().__init__(**kw)
        # Exact/True AP loss (not differentiable)
        if nq == 0:
            nq = 'sklearn'  # special case
        try:
            self.compute_AP = eval('self.compute_true_AP_' + nq)
        except:
            raise ValueError("Unknown mode %s for AP loss" % nq)

    @staticmethod
    def compute_true_AP_sklearn(scores, labels):
        def compute_AP(label, score):
            return average_precision_score(label, score)

        aps = scores.new_zeros((scores.shape[0], scores.shape[1]))
        label_np = labels.cpu().numpy().astype(bool)
        scores_np = scores.cpu().numpy()
        for bi in range(scores_np.shape[0]):
            for i in range(scores_np.shape[1]):
                labels = label_np[bi, i, :]
                if labels.sum() < 1:
                    continue
                aps[bi, i] = compute_AP(labels, scores_np[bi, i, :])
        return aps

    @staticmethod
    def compute_true_AP_torch(scores, labels):
        assert scores.shape == labels.shape
        B, N, M = labels.shape
        dev = labels.device
        with torch.no_grad():
            # sort scores
            _, order = scores.sort(dim=-1, descending=True)
            # sort labels accordingly
            labels = labels[torch.arange(B, device=dev)[:, None, None].expand(order.shape),
                            torch.arange(N, device=dev)[None, :, None].expand(order.shape),
                            order]
            # compute number of positives per query
            npos = labels.sum(dim=-1)
            assert torch.all(torch.isclose(npos, npos[0, 0])
                             ), "only implemented for constant number of positives per query"
            npos = int(npos[0, 0])
            # compute precision at each recall point
            posrank = labels.nonzero()[:, -1].view(B, N, npos)
            recall = torch.arange(1, 1 + npos, dtype=torch.float32, device=dev)[None, None, :].expand(B, N, npos)
            precision = recall / (1 + posrank).float()
            # average precision values at all recall points
            aps = precision.mean(dim=-1)

        return aps

    def loss(self, desc1, desc2, valid_matches=None, euc=False):  # if matches is None, positives are the diagonal
        B, N1, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2

        scores = get_similarities(desc1, desc2, euc)

        labels = torch.zeros([B, N1, N2], dtype=scores.dtype, device=scores.device)

        # allow all diagonal positives and only mask afterwards
        labels.diagonal(dim1=-2, dim2=-1)[...] = 1.
        apscore = self.compute_AP(scores, labels)
        if valid_matches is not None:
            apscore = apscore[valid_matches]
        return apscore


class MatchingLoss (Criterion, MultiLoss):
    """ 
    Matching loss per image 
    only compare pixels inside an image but not in the whole batch as what would be done usually
    """

    def __init__(self, criterion, withconf=False, use_pts3d=False, negatives_padding=0, blocksize=4096):
        super().__init__(criterion)
        self.negatives_padding = negatives_padding
        self.use_pts3d = use_pts3d
        self.blocksize = blocksize
        self.withconf = withconf

    def add_negatives(self, outdesc2, desc2, batchid, x2, y2):
        if self.negatives_padding:
            B, H, W, D = desc2.shape
            negatives = torch.ones([B, H, W], device=desc2.device, dtype=bool)
            negatives[batchid, y2, x2] = False
            sel = negatives & (negatives.view([B, -1]).cumsum(dim=-1).view(B, H, W)
                               <= self.negatives_padding)  # take the N-first negatives
            outdesc2 = torch.cat([outdesc2, desc2[sel].view([B, -1, D])], dim=1)
        return outdesc2

    def get_confs(self, pred1, pred2, sel1, sel2):
        if self.withconf:
            if self.use_pts3d:
                outconfs1 = pred1['conf'][sel1]
                outconfs2 = pred2['conf'][sel2]
            else:
                outconfs1 = pred1['desc_conf'][sel1]
                outconfs2 = pred2['desc_conf'][sel2]
        else:
            outconfs1 = outconfs2 = None
        return outconfs1, outconfs2

    def get_descs(self, pred1, pred2):
        if self.use_pts3d:
            desc1, desc2 = pred1['pts3d'], pred2['pts3d_in_other_view']
        else:
            desc1, desc2 = pred1['desc'], pred2['desc']
        return desc1, desc2

    def get_matching_descs(self, gt1, gt2, pred1, pred2, **kw):
        outdesc1 = outdesc2 = outconfs1 = outconfs2 = None
        # Recover descs, GT corres and valid mask
        desc1, desc2 = self.get_descs(pred1, pred2)

        (x1, y1), (x2, y2) = gt1['corres'].unbind(-1), gt2['corres'].unbind(-1)
        valid_matches = gt1['valid_corres']

        # Select descs that have GT matches
        B, N = x1.shape
        batchid = torch.arange(B)[:, None].repeat(1, N)  # B, N
        outdesc1, outdesc2 = desc1[batchid, y1, x1], desc2[batchid, y2, x2]  # B, N, D

        # Padd with unused negatives
        outdesc2 = self.add_negatives(outdesc2, desc2, batchid, x2, y2)

        # Gather confs if needed
        sel1 = batchid, y1, x1
        sel2 = batchid, y2, x2
        outconfs1, outconfs2 = self.get_confs(pred1, pred2, sel1, sel2)

        return outdesc1, outdesc2, outconfs1, outconfs2, valid_matches, {'use_euclidean_dist': self.use_pts3d}

    def blockwise_criterion(self, descs1, descs2, confs1, confs2, valid_matches, euc, rng=np.random, shuffle=True):
        loss = None
        details = {}
        B, N, D = descs1.shape

        if N <= self.blocksize:  # Blocks are larger than provided descs, compute regular loss
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
        else:  # Compute criterion on the blockdiagonal only, after shuffling
            # Shuffle if necessary
            matches_perm = slice(None)
            if shuffle:
                matches_perm = np.stack([rng.choice(range(N), size=N, replace=False) for _ in range(B)])
                batchid = torch.tile(torch.arange(B), (N, 1)).T
                matches_perm = batchid, matches_perm

            descs1 = descs1[matches_perm]
            descs2 = descs2[matches_perm]
            valid_matches = valid_matches[matches_perm]

            assert N % self.blocksize == 0, "Error, can't chunk block-diagonal, please check blocksize"
            n_chunks = N // self.blocksize
            descs1 = descs1.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            descs2 = descs2.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            valid_matches = valid_matches.view([B * n_chunks, self.blocksize])
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
            if self.withconf:
                confs1, confs2 = map(lambda x: x[matches_perm], (confs1, confs2))  # apply perm to confidences if needed

        if self.withconf:
            # split confidences between positives/negatives for loss computation
            details['conf_pos'] = map(lambda x: x[valid_matches.view(B, -1)], (confs1, confs2))
            details['conf_neg'] = map(lambda x: x[~valid_matches.view(B, -1)], (confs1, confs2))
            details['Conf1_std'] = confs1.std()
            details['Conf2_std'] = confs2.std()

        return loss, details

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # Gather preds and GT
        descs1, descs2, confs1, confs2, valid_matches, monitoring = self.get_matching_descs(
            gt1, gt2, pred1, pred2, **kw)

        # loss on matches
        loss, details = self.blockwise_criterion(descs1, descs2, confs1, confs2,
                                                 valid_matches, euc=monitoring.pop('use_euclidean_dist', False))

        details[type(self).__name__] = float(loss.mean())
        return loss, (details | monitoring)


def check_convergence(
    iter,
    rel_error_threshold,
    delta_norm_threshold,
    old_cost,
    new_cost,
    delta,
    verbose=False,
):
    # step,
    # self.cfg["rel_error"],
    # self.cfg["delta_norm"],
    # old_cost,
    # new_cost,
    # tau_ij_sim3,

    cost_diff = old_cost - new_cost
    rel_dec = math.fabs(cost_diff / old_cost)
    delta_norm = torch.linalg.norm(delta)

    converged = rel_dec < rel_error_threshold or delta_norm < delta_norm_threshold
    if verbose:
        print(
            f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}"
        )

    # print(f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}")
    return converged

class RenderLoss(Criterion, MultiLoss):
    """Loss for comparing rendered depth maps from predicted 3D points to ground truth depth maps."""
    # l_vec: light direction
    # n_vec: normal vector
    # h_vec: helf vector = v+l/v+l

    def __init__(self, criterion, dist_clip=None):
        super().__init__(criterion)
        self.dist_clip = dist_clip  # Optional distance clipping for robustness

    def get_name(self):
        return f'RenderLoss({self.criterion})'

    import torch

    def compute_pixel_normals(self, pts3d, valid_mask=None):
        """
        Compute surface normals for each pixel in a 3D point cloud.
        Args:
            pts3d: torch.Tensor of shape [B, H, W, 3], 3D coordinates (x, y, z) for each pixel
            valid_mask: torch.Tensor of shape [B, H, W], boolean mask for valid pixels (optional)
        Returns:
            normals: torch.Tensor of shape [B, H, W, 3], unit normal vectors for each pixel
            valid_normals: torch.Tensor of shape [B, H, W], boolean mask for valid normals
        """
        B, H, W, _ = pts3d.shape

        # Initialize validity mask if not provided
        if valid_mask is None:
            valid_mask = torch.ones(B, H, W, dtype=torch.bool, device=pts3d.device)

        # Initialize output normals and validity mask
        normals = torch.zeros_like(pts3d)  # Shape: [B, H, W, 3]
        valid_normals = torch.zeros_like(valid_mask)  # Shape: [B, H, W]

        # Compute vectors to right and down neighbors
        vec_right = pts3d[:, :, 1:, :] - pts3d[:, :, :-1, :]  # Shape: [B, H, W-1, 3]
        vec_down = pts3d[:, 1:, :, :] - pts3d[:, :-1, :, :]  # Shape: [B, H-1, W, 3]

        # Compute valid masks for neighbors
        valid_right = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]  # Shape: [B, H, W-1]
        valid_down = valid_mask[:, :-1, :] & valid_mask[:, 1:, :]  # Shape: [B, H-1, W]

        # Pad vectors and masks to match original shape
        vec_right = torch.nn.functional.pad(vec_right, (0, 0, 0, 1), mode='constant', value=0)  # Shape: [B, H, W, 3]
        vec_down = torch.nn.functional.pad(vec_down, (0, 0, 0, 0, 0, 1), mode='constant', value=0)  # Shape: [B, H, W, 3]
        valid_right = torch.nn.functional.pad(valid_right, (0, 1), mode='constant', value=False)  # Shape: [B, H, W]
        valid_down = torch.nn.functional.pad(valid_down, (0, 0, 0, 1), mode='constant', value=False)  # Shape: [B, H, W]

        # Compute cross product: normal = vec_right x vec_down
        normals = torch.cross(vec_right, vec_down, dim=-1)  # Shape: [B, H, W, 3]

        # Normalize normals to unit vectors
        norm = torch.norm(normals, dim=-1, keepdim=True)
        normals = normals / torch.clamp(norm, min=1e-8)  # Avoid division by zero

        # Valid normals require valid pixel and both neighbors
        valid_normals = valid_mask & valid_right & valid_down

        # Set normals to zero for invalid pixels
        normals = normals * valid_normals.unsqueeze(-1)

        return normals, valid_normals

    def point_to_ray_dist(self, X, jacobian=False):
        b = X.shape[:-1]

        def point_to_dist(X):
            d = torch.linalg.norm(X, dim=-1, keepdim=True)
            return d

        d = point_to_dist(X)
        d_inv = 1.0 / d
        r = d_inv * X
        rd = torch.cat((r, d), dim=-1)  # Dim 4
        if not jacobian:
            return rd
        else:
            d_inv_2 = d_inv ** 2
            I = torch.eye(3, device=X.device, dtype=X.dtype).repeat(*b, 1, 1)
            dr_dX = d_inv.unsqueeze(-1) * (
                    I - d_inv_2.unsqueeze(-1) * (X.unsqueeze(-1) @ X.unsqueeze(-2))
            )
            dd_dX = r.unsqueeze(-2)
            drd_dX = torch.cat((dr_dX, dd_dX), dim=-2)
            return rd, drd_dX

    def act_Sim3(self, X: lietorch.Sim3, pC: torch.Tensor, jacobian=False):
        pW = X.act(pC)
        if not jacobian:
            return pW
        dpC_dt = torch.eye(3, device=pW.device).repeat(*pW.shape[:-1], 1, 1)
        dpC_dR = -skew_sym(pW)
        dpc_ds = pW.reshape(*pW.shape[:-1], -1, 1)
        return pW, torch.cat([dpC_dt, dpC_dR, dpc_ds], dim=-1)  # view(-1, mdim)

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r

        def huber(r, k=1.345):
            unit = torch.ones((1), dtype=r.dtype, device=r.device)
            r_abs = torch.abs(r)
            mask = r_abs < k
            w = torch.where(mask, unit, k / r_abs)
            return w

        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        # L = torch.linalg.cholesky(H, upper=False)
        # tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        try:
            L = torch.linalg.cholesky(H, upper=False)
            tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)
        except RuntimeError as e:
            # Fallback: Add damping (Levenberg-Marquardt style) or use torch.linalg.solve
            damping = 1e-6 * torch.eye(H.shape[0], device=H.device)
            H_damped = H + damping
            try:
                L = torch.linalg.cholesky(H_damped, upper=False)
                tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)
            except RuntimeError:
                # Final fallback: Use torch.linalg.solve
                tau_j = torch.linalg.solve(H_damped, g).view(1, -1)
                print(f"Warning: Cholesky failed, used damped linear solve. Error: {str(e)}")

        return tau_j, cost

    def check_convergence(self, step, rel_error, delta_norm, old_cost, new_cost, delta):
        """
        Check convergence of optimization for batched inputs.
        """
        if step > 0 and torch.all(torch.abs(old_cost - new_cost) < rel_error):
            return True
        if torch.all(torch.norm(delta, dim=1) < delta_norm):
            return True
        return False

    def optimize_camera_pose(self, pts3d1, pts3d2, conf1, conf2, conf_thresh=1.0, cfg=None):
        """
        Optimize the relative pose between two cameras using 3D points for batched inputs.
        Args:
            pts3d1: torch.Tensor of shape [B, H, W, 3], 3D points in camera 1 frame
            pts3d2: torch.Tensor of shape [B, H, W, 3], 3D points in camera 2 frame
            conf1: torch.Tensor of shape [B, H, W], confidence scores for pts3d1
            conf2: torch.Tensor of shape [B, H, W], confidence scores for pts3d2
            conf_thresh: float, confidence threshold
            cfg: dict, configuration parameters
        Returns:
            T_WC1: torch.Tensor of shape [B, 4, 4], optimized pose of camera 1
            T_C2C1: torch.Tensor of shape [B, 4, 4], optimized relative pose
        """
        B, H, W, _ = pts3d1.shape
        pts3d1 = pts3d1.view(B, -1, 3)  # [B, H*W, 3]
        pts3d2 = pts3d2.view(B, -1, 3)  # [B, H*W, 3]
        conf1 = conf1.view(B, -1)  # [B, H*W]
        conf2 = conf2.view(B, -1)  # [B, H*W]

        if cfg is None:
            cfg = {
                "sigma_ray": 0.003,
                "sigma_dist": 1,
                "max_iters": 50,
                "rel_error": 1e-3,
                "delta_norm": 1e-3
            }

        # Filter points based on confidence
        valid1 = conf1 > conf_thresh
        valid2 = conf2 > conf_thresh
        valid_opt = valid1 & valid2

        # Apply mask to points and confidences
        B, N = pts3d1.shape[:2]
        pts3d1 = pts3d1 * valid_opt[:, :, None]
        pts3d2 = pts3d2 * valid_opt[:, :, None]
        conf = torch.min(conf1, conf2) * valid_opt

        # each batch is one OK:
        T_WC1_list = []
        T_C2C1_list = []
        for i in range(B):
            last_error = 0
            T_WC1 = lietorch.Sim3.Identity(1, device=pts3d1.device)
            T_WC2 = lietorch.Sim3.Identity(1, device=pts3d1.device)
            T_C2C1 = lietorch.Sim3.Identity(1, device=pts3d1.device)
            # T_WC1 = torch.eye(4, device=pts3d1.device).unsqueeze(0).expand(B, 4, 4)
            # T_WC2 = torch.eye(4, device=pts3d1.device).unsqueeze(0).expand(B, 4, 4)
            # T_C2C1 = torch.eye(4, device=pts3d1.device).unsqueeze(0).expand(B, 4, 4)

            # Initialize information weights
            sqrt_info_ray = 1 / cfg["sigma_ray"] * valid_opt * torch.sqrt(conf)
            sqrt_info_dist = 1 / cfg["sigma_dist"] * valid_opt * torch.sqrt(conf)
            sqrt_info = torch.cat((sqrt_info_ray.unsqueeze(-1).repeat(1, 1, 3), sqrt_info_dist.unsqueeze(-1)), dim=2)

            # Precalculate ray distances for camera 2
            rd_2, _ = self.point_to_ray_dist(pts3d2[i], jacobian=False)

            old_cost = torch.full((B,), float("inf"), device=pts3d1.device)
            for step in range(cfg["max_iters"]):
                # Transform points from camera 1 to camera 2
                pts1_C2, dpts1_C2_dT_C2C1 = self.act_SE3(T_C2C1, pts3d1[i], jacobian=True)
                rd_1_C2, drd_1_C2_dpts1_C2 = self.point_to_ray_dist(pts1_C2, jacobian=True)

                # Compute residuals
                residuals = rd_2 - rd_1_C2

                # Compute Jacobian
                J = -drd_1_C2_dpts1_C2 @ dpts1_C2_dT_C2C1

                # Solve for update
                tau_ij_sim3, new_cost = self.solve(sqrt_info, residuals, J)
                T_C2C1 = T_C2C1.retr(tau_ij_sim3)
                T_C2C1_list.append(T_C2C1)

                if check_convergence(
                        step,
                        self.cfg["rel_error"],
                        self.cfg["delta_norm"],
                        old_cost,
                        new_cost,
                        tau_ij_sim3,
                ):
                    break
                old_cost = new_cost

                if step == self.cfg["max_iters"] - 1:
                    print(f"max iters reached {last_error}")

            T_WC1 = T_WC2 @ T_C2C1
            T_WC1_list.append(T_WC1)

        T_WC1_list = torch.tensor(T_WC1_list)
        T_C2C1_list = torch.tensor(T_C2C1_list)
        return T_WC1_list, T_C2C1


    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        """Compute loss between rendered and ground truth depth maps."""
        # Extract ground truth and predicted 3D points
        gt_img1 = gt1['img'] # B, 3, H, w
        gt_img2 = gt2['img'] # B, 3, H, W

        pred_albedo1 = pred1['albedo']
        pred_albedo2 = pred2['albedo']

        pred_specular1 = pred1['specular']
        pred_specular2 = pred2['specular']

        pred1_pts3d = pred1['pts3d']
        pred2_pts3d = pred2['pts3d_in_other_view']

        pred_depth1 = pred1_pts3d[:, :, :, -1]
        pred_depth2 = pred2_pts3d[:, :, :, -1]

        conf1 = pred1['conf']
        conf2 = pred2['conf']

        Q1 = pred1['desc_conf']
        Q2 = pred2['desc_conf']

        ##---------------diffuse Term-----------------##
        # calculate normal vector

        valid_mask1 = gt1['valid_mask']
        valid_mask2 = gt2['valid_mask']

        n1_vec, valid_n1 = self.compute_pixel_normals(pred1_pts3d, valid_mask1)
        n2_vec, valid_n2 = self.compute_pixel_normals(pred2_pts3d, valid_mask2)

        # calculate light vector: need camera pose
        T_WC1, T_C2C1 = self.optimize_camera_pose(pred1_pts3d, pred2_pts3d, conf1, conf2)
        T_WC2 = T_WC1 * torch.linalg.inv(T_C2C1)
        l1_vec = T_WC1[:3, 3]/torch.linalg.norm(T_WC1[:3, 3])
        l2_vec = T_WC2[:3, 3]/torch.linalg.norm(T_WC2[:3, 3])

        # diffuse term
        cos_theta1 = torch.dot(n1_vec, l1_vec)
        cos_theta2 = torch.dot(n2_vec, l2_vec)
        Ld1 = pred_albedo1 * 1.0 / (pred_depth1**2) * cos_theta1
        Ld2 = pred_albedo2 * 1.0 / (pred_depth2**2) * cos_theta2

        ##---------------specular term----------------##
        Ls1 = pred_specular1 * 1.0 / (pred_depth1**2)
        Ls2 = pred_specular2 * 1.0 / (pred_depth2**2)

        ##---------------rendering RGB----------------##
        color1 = Ld1 + Ls1
        color2 = Ld2 + Ls2

        ##--------------compute loss------------------##

        # gt_pts1 = gt1['pts3d']  # Shape: (B, H, W, 3)
        # gt_pts2 = gt2['pts3d']
        # pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        # pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)
        #
        # # Extract camera intrinsics and validity masks
        # camera_intrinsics1 = gt1['camera_intrinsics']  # Shape: (B, 3, 3)
        # camera_intrinsics2 = gt2['camera_intrinsics']
        # valid1 = gt1['valid_mask'].clone()
        # valid2 = gt2['valid_mask'].clone()
        #
        # # Render depth maps
        # pr_depth1, valid1 = self.render_depth_map(pr_pts1, camera_intrinsics1, valid1)
        # pr_depth2, valid2 = self.render_depth_map(pr_pts2, camera_intrinsics2, valid2)
        # gt_depth1, valid1 = self.render_depth_map(gt_pts1, camera_intrinsics1, valid1)
        # gt_depth2, valid2 = self.render_depth_map(gt_pts2, camera_intrinsics2, valid2)

        # Compute loss for each view
        loss1 = self.criterion(color1, gt_img1)
        loss2 = self.criterion(color2, gt_img2)

        valid1 = valid_n1 & valid_mask1
        valid2 = valid_n2 & valid_mask2
        # Combine losses using Sum
        details = {
            f'{type(self).__name__}_depth_1': float(loss1.mean()),
            f'{type(self).__name__}_depth_2': float(loss2.mean())
        }
        return Sum((loss1, valid1), (loss2, valid2)), details


class ConfMatchingLoss(ConfLoss):
    """ Weight matching by learned confidence. Same as ConfLoss but for a matching criterion
        Assuming the input matching_loss is a match-level loss.
    """

    def __init__(self, pixel_loss, alpha=1., confmode='prod', neg_conf_loss_quantile=False):
        super().__init__(pixel_loss, alpha)
        self.pixel_loss.withconf = True
        self.confmode = confmode
        self.neg_conf_loss_quantile = neg_conf_loss_quantile

    def aggregate_confs(self, confs1, confs2):  # get the confidences resulting from the two view predictions
        if self.confmode == 'prod':
            confs = confs1 * confs2 if confs1 is not None and confs2 is not None else 1.
        elif self.confmode == 'mean':
            confs = .5 * (confs1 + confs2) if confs1 is not None and confs2 is not None else 1.
        else:
            raise ValueError(f"Unknown conf mode {self.confmode}")
        return confs

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        loss, details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        # Recover confidences for positive and negative samples
        conf1_pos, conf2_pos = details.pop('conf_pos')
        conf1_neg, conf2_neg = details.pop('conf_neg')
        conf_pos = self.aggregate_confs(conf1_pos, conf2_pos)

        # weight Matching loss by confidence on positives
        conf_pos, log_conf_pos = self.get_conf_log(conf_pos)
        conf_loss = loss * conf_pos - self.alpha * log_conf_pos
        # average + nan protection (in case of no valid pixels at all)
        conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
        # Add negative confs loss to give some supervision signal to confidences for pixels that are not matched in GT
        if self.neg_conf_loss_quantile:
            conf_neg = torch.cat([conf1_neg, conf2_neg])
            conf_neg, log_conf_neg = self.get_conf_log(conf_neg)

            # recover quantile that will be used for negatives loss value assignment
            neg_loss_value = torch.quantile(loss, self.neg_conf_loss_quantile).detach()
            neg_loss = neg_loss_value * conf_neg - self.alpha * log_conf_neg

            neg_loss = neg_loss.mean() if neg_loss.numel() > 0 else 0
            conf_loss = conf_loss + neg_loss

        return conf_loss, dict(matching_conf_loss=float(conf_loss), **details)
