import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib

def norm(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())


if __name__ == '__main__':
    metric_root = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_dong/MASt3R_1023_finetune_w_qualitative_synthetic"
    metric_gt_list = sorted(glob(os.path.join(metric_root, '*_gt.png')))
    metric_pred_list = sorted(glob(os.path.join(metric_root, '*_depth.png')))

    save_folder = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_dong/MASt3R_1023_qualitative_synthetic_align"
    os.makedirs(save_folder, exist_ok=True)
    for metric_pred, metric_gt in zip(metric_pred_list, metric_gt_list):
        img_name = metric_pred.split('/')[-1]
        gt_name = metric_gt.split('/')[-1]

        metric_pred_img = Image.open(metric_pred)
        metric_pred_img = metric_pred_img.convert('L')
        metric_pred_array = np.array(metric_pred_img, dtype=np.float32)

        metric_gt_img = Image.open(metric_gt)
        metric_gt_img = metric_gt_img.convert('L')
        metric_gt_array = np.array(metric_gt_img, dtype=np.float32)

        relative_pred_array = norm(metric_pred_array)
        jet_cmap = matplotlib.cm.get_cmap('jet')
        pred_image_rgb = jet_cmap(relative_pred_array)[:, :, :3]  # 取 RGB 通道，忽略 alpha
        pred_image_rgb = (pred_image_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型

        # 使用 PIL 保存图像
        pil_image = Image.fromarray(pred_image_rgb)
        pil_image.save(os.path.join(save_folder, img_name))

        relative_gt_array = norm(metric_gt_array)
        jet_cmap = matplotlib.cm.get_cmap('jet')
        relative_gt_rgb = jet_cmap(relative_gt_array)[:, :, :3]  # 取 RGB 通道，忽略 alpha
        relative_gt_rgb = (relative_gt_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型

        # 使用 PIL 保存图像
        pil_image = Image.fromarray(relative_gt_rgb)
        pil_image.save(os.path.join(save_folder, gt_name))


