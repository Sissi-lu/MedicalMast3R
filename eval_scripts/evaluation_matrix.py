
import torch
import cv2
from skimage import metrics
import numpy as np
from eval_scripts.robustmvd_metrics import m_rel_ae, thresh_inliers

def eval_depth_numpy(pred, target, msk):
    # Convert torch tensors to numpy arrays if they aren't already
    if hasattr(pred, 'numpy'):
        pred = pred.numpy()
    if hasattr(target, 'numpy'):
        target = target.numpy()
    if msk is not None and hasattr(msk, 'numpy'):
        msk = msk.numpy()

    assert pred.shape == target.shape

    mvd_absrel = m_rel_ae(gt=target, pred=pred, mask=msk, output_scaling_factor=100.0)
    mvd_inliers103 = thresh_inliers(gt=target, pred=pred, thresh=1.03, mask=msk, output_scaling_factor=100.0)

    # Masking operation
    if msk is not None:
        pred = pred[msk]
        target = target[msk]
    else:
        pred = pred[target != 0.0]
        target = target[target != 0.0]

    # Calculate threshold metrics
    thresh = np.maximum((target / pred), (pred / target))

    d05 = np.sum(thresh < 1.25**0.5) / len(thresh)
    d1 = np.sum(thresh < 1.25) / len(thresh)
    d2 = np.sum(thresh < 1.25 ** 2) / len(thresh)
    d3 = np.sum(thresh < 1.25 ** 3) / len(thresh)

    # Differences
    diff = pred - target
    diff_log = np.log(pred + 1e-6) - np.log(target + 1e-6)

    # Metrics
    abs_rel = np.mean(np.abs(diff) / (target + 1e-6))
    sq_rel = np.mean(np.power(diff, 2) / (target + 1e-6))

    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    rmse_log = np.sqrt(np.mean(np.power(diff_log, 2)))

    log10 = np.mean(np.abs(np.log10(pred + 1e-6) - np.log10(target + 1e-6)))
    silog = np.sqrt(np.mean(np.power(diff_log, 2)) - 0.5 * np.power(np.mean(diff_log), 2))


    return {
        'robust_mvd_absrel': float(mvd_absrel),
        'inliers103': float(mvd_inliers103),
        'd05': float(d05),
        'd1': float(d1),  # Convert to native Python float
        'd2': float(d2),
        'd3': float(d3),
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'log10': float(log10),
        'silog': float(silog)
    }


def eval_depth(pred, target, msk):
    assert pred.shape == target.shape


    if msk is not None:
        pred = pred[msk]
        target = target[msk]
    else:
        pred = pred[target != 0.0]
        target = target[target != 0.0]
        # pred = pred.view(-1)
        # target = target.view(-1)

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / (target+1e-6))
    sq_rel = torch.mean(torch.pow(diff, 2) / (target+1e-6))

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(),
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}

def eval_fps(pred, pred_after):
    # print(type(pred))
    # print(type(pred_after))
    # print(pred.shape)
    # print(pred_after.shape)

    pred = pred.detach().cpu().numpy().astype('uint8')
    pred_after = pred_after.detach().cpu().numpy().astype('uint8')

    # Histogram-Based Approaches
    hist_img1 = cv2.calcHist([pred], [0], None, [256], [0, 256])
    # hist_img1[255, 255, 255] = 0  # ignore all white pixels
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([pred_after], [0], None, [256], [0, 256])
    # hist_img2[255, 255, 255] = 0  # ignore all white pixels
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # Find the metric value
    HB_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

    # SSIM
    ssim_score = metrics.structural_similarity(pred, pred_after, full=True)[0]


    return {'HB_score': HB_val, "ssim_score": ssim_score}

def eval_focal(pred, gt):
    diff = np.abs(pred - gt)
    abs_rel = diff / gt
    return abs_rel