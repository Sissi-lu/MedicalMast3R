import os
import cv2
import numpy as np

folder_path = '/data_new/luxiaoxi/dongbingwen/pred2/marigold/part2/left'  # 图片文件夹
# name = '00423'  # 查找文件名包含的字符串
mask_path = '/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/part2/left/valid_region_mask'  # mask 文件路径
mask_color = (127, 0, 0)  # BGR格式，初始为蓝紫色

# folder_path = '/data_new/luxiaoxi/dongbingwen/pred/vggt_test_zeroshot/s7_processed/left/imgs'   # 图片文件夹
# name = '00330_depth'                  # 查找文件名包含的字符串
# mask_path = '/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/s7_processed/left/valid_region_mask/00330.png'   # mask 文件路径
# mask_color = (127, 0, 0)        # BGR格式，初始为蓝紫色



# save_folder = '/data_new/luxiaoxi/dongbingwen/pred/masked'
# os.makedirs(save_folder, exist_ok=True)
# ========== 遍历文件夹 ==========
for filename in os.listdir(folder_path):
    # if name in filename and filename.lower().endswith('.png'):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(mask_path, filename.split('_')[0]+'.png'), cv2.IMREAD_GRAYSCALE)

        mask_binary = (mask > 0).astype(np.uint8)

        # 确保图片与 mask 尺寸一致
        if img.shape[:2] != mask_binary.shape:
            img_resized = cv2.resize(img, (mask_binary.shape[1], mask_binary.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        # 应用 mask
        # img_resized = img.copy()
        img_resized[mask_binary == 0] = mask_color

        save_name = os.path.splitext(filename)[0] + '_mask.png'
        save_path = os.path.join(folder_path, save_name)
        cv2.imwrite(save_path, img_resized)
        print(f"保存: {save_path}")
