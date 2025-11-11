import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib
def align(gt, pred, eps=1e-6):
    Y = gt
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    scale, shift = np.linalg.lstsq(A, Y, rcond=None)[0]
    return scale, shift
    # output = shift + scale * pred
    # return np.maximum(output, eps)

if __name__ == '__main__':
    relative_root = '/data_new/luxiaoxi/dongbingwen/pred/zeodepth_test_finetune'
    relative_gt_list = sorted(glob(os.path.join(relative_root, '*_gt_depth.png')))

    metric_root = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_dong/MASt3R_1023_finetune_w_qualitative_synthetic"
    metric_gt_list = sorted(glob(os.path.join(metric_root, '*_gt.png')))

    save_folder = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_dong/MASt3R_1023_qualitative_synthetic_align"
    os.makedirs(save_folder, exist_ok=True)
    for relative_gt, metric_gt in zip(relative_gt_list, metric_gt_list):
        relative_gt_img = Image.open(relative_gt)
        relative_gt_img = relative_gt_img.convert('L')
        relative_gt_array = np.array(relative_gt_img, dtype=np.float32)

        mask = relative_gt_array != 14.0

        metric_gt_img = Image.open(metric_gt)
        metric_gt_img = metric_gt_img.convert('L')
        metric_gt_array = np.array(metric_gt_img, dtype=np.float32)

        relative_gt_mask = relative_gt_array[mask]
        metric_gt_mask = metric_gt_array[mask]
        scale, shift = align(relative_gt_mask, metric_gt_mask)

        pred_metric = metric_gt.replace('gt', 'depth')

        img_name = pred_metric.split('/')[-1]
        pred_metric_img = Image.open(pred_metric)
        pred_metric_img = pred_metric_img.convert('L')
        pred_metric_array = np.array(pred_metric_img, dtype=np.float32)

        align_metric_gt_array = metric_gt_array
        align_metric_gt_array[mask] = shift + scale * align_metric_gt_array[mask]


        align_metric_gt_array_norm = (align_metric_gt_array - np.min(align_metric_gt_array)) / (
                    np.max(align_metric_gt_array) - np.min(align_metric_gt_array))
        jet_cmap = matplotlib.cm.get_cmap('jet')
        gt_image_rgb = jet_cmap(align_metric_gt_array_norm)[:, :, :3]  # 取 RGB 通道，忽略 alpha
        gt_image_rgb = (gt_image_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型

        # 使用 PIL 保存图像
        pil_image = Image.fromarray(gt_image_rgb)
        pil_image.save(os.path.join(save_folder, img_name.split('.')[0] + '_align_gt.png'))


        pred_relative_array = pred_metric_array.copy()
        pred_relative_array[mask] = shift + pred_metric_array[mask] * scale

        pred_relative_array_norm = (pred_relative_array - np.min(pred_relative_array))/(np.max(pred_relative_array) - np.min(pred_relative_array))
        jet_cmap = matplotlib.cm.get_cmap('jet')
        pred_image_rgb = jet_cmap(pred_relative_array_norm)[:, :, :3]  # 取 RGB 通道，忽略 alpha
        pred_image_rgb = (pred_image_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型

        # 使用 PIL 保存图像
        pil_image = Image.fromarray(pred_image_rgb)
        pil_image.save(os.path.join(save_folder, img_name))

