from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def reverse(depth_map):
    result = depth_map.copy()
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    if d_max != d_min:
        result = d_max - (depth_map - d_min)
    else:
        return result
    return result


def apply_depth_with_mask_and_reverse_background(
    depth_path: str,
    output_path: str = None
):
    """
    - depth_path: grayscale depth map (smaller value = closer, or larger = closer, doesn't matter)
    - mask_path:  binary or grayscale mask (non-zero = foreground/object, zero = background)
    - Only background (mask == 0) gets depth reversal
    """
    # 1. Load depth and mask as float arrays
    depth_img = Image.open(depth_path).convert('L')
    depth = np.array(depth_img)
    mask  =  depth == 14.0

    # 2. Create a unified depth array:
    #    - Inside mask  → keep original depth
    #    - Outside mask → reverse depth (so background becomes "far")
    final_depth = depth.copy()

    # Background pixels (mask == 0 or close to 0)
    bg = ~mask  # threshold, works for both binary and soft masks

    if bg.any():
        bg_depth = depth[bg]
        d_min = depth.min()
        d_max = depth.max()

        if d_max > d_min:
            # Reverse only background
            reversed_bg = d_max - (bg_depth - d_min)
            final_depth[bg] = reversed_bg
        # else: constant depth → nothing to reverse

    # 3. Normalize final_depth to [0, 1] for colormap
    d_min = final_depth.min()
    d_max = final_depth.max()
    if d_max > d_min:
        normalized = (final_depth - d_min) / (d_max - d_min)
    else:
        normalized = np.zeros_like(final_depth)

    # 4. Apply Jet colormap
    cmap = plt.get_cmap('jet')
    colored = cmap(normalized)[..., :3]                  # (H, W, 3) float32 in [0,1]
    colored_uint8 = (colored * 255).astype(np.uint8)

    # 5. Save result
    result = Image.fromarray(colored_uint8, mode='RGB')

    if output_path is None:
        base, ext = os.path.splitext(depth_path)
        output_path = f"{base}_jet_maskreversed{ext}"

    result.save(output_path)
    print(f"Saved → {output_path}")
    return result

if __name__ == '__main__':
    data_dir = "/data_new/luxiaoxi/dongbingwen/pred_lxx/dust3r"
    save_dir = "/data_new/luxiaoxi/dongbingwen/pred_lxx/dust3r_reverse"
    os.makedirs(save_dir, exist_ok=True)

    img_list = sorted(os.listdir(data_dir))
    for img_name in img_list:
        img_path = os.path.join(data_dir, img_name)
        save_path = os.path.join(save_dir, img_name)
        jet_img = apply_depth_with_mask_and_reverse_background(img_path, save_path)

