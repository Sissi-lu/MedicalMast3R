import os
from PIL import Image
import numpy as np
import matplotlib.cm as cm

data_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed_image/right_depth"
save_dir = data_dir.replace('right', 'right_colored')
os.makedirs(save_dir, exist_ok=True)

img_list = sorted(os.listdir(data_dir))
for img_name in img_list:
    img = Image.open(os.path.join(data_dir, img_name))
    img_array = np.array(img)[:, :, 0]

    # 归一化到 [0, 1]（jet 色图需要归一化输入）
    img_normalized = (img_array - np.min(img_array) )/ (np.max(img_array) - np.min(img_array))

    # 应用 jet 色图
    jet_cmap = cm.get_cmap('jet')
    img_rgb = jet_cmap(img_normalized)[:, :, :3]  # 取 RGB 通道，忽略 alpha
    img_rgb = (img_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型

    # 转换为 PIL 图像
    pil_image = Image.fromarray(img_rgb)

    # 保存 RGB 图像
    pil_image.save(os.path.join(save_dir, img_name))