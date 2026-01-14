from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def gray_to_jet_matplotlib(input_path, output_path=None):
    img = Image.open(input_path).convert('L')
    arr = np.array(img)
    arr_norm = arr.astype(float) / arr.max() if arr.max() > 0 else arr.astype(float)

    cmap = plt.get_cmap('jet')
    colored = cmap(arr_norm)[..., :3]  # drop alpha channel
    colored = (colored * 255).astype(np.uint8)

    jet_img = Image.fromarray(colored, mode='RGB')
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_jet{ext}"
    jet_img.save(output_path)
    return jet_img

if __name__ == '__main__':
    data_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/part2/left/depth"
    save_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/part2/left/color_depth"
    os.makedirs(save_dir, exist_ok=True)

    img_list = sorted(os.listdir(data_dir))
    for img_name in img_list:
        img_path = os.path.join(data_dir, img_name)
        out_path = os.path.join(save_dir, img_name)
        gray_to_jet_matplotlib(img_path, out_path)