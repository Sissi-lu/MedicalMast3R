import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────
data_dir    = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/part4"
output_dir  = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_real/MASt3R_0124_real"
name        = "06732"

right_path  = os.path.join(data_dir, "right", f"{name}.jpg")
render_path = os.path.join(output_dir, f"reproj_left_from_right_{name}.png")  # adjust if needed

# ────────────────────────────────────────────────
def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range=255.0) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from scipy.ndimage import gaussian_filter
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1**2, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, 1.5) - mu2_sq
    sigma12   = gaussian_filter(img1*img2, 1.5) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────
print("Loading images...")

right_img  = load_image_rgb(right_path)
render_img = load_image_rgb(render_path)

print(f"Right  shape: {right_img.shape}")
print(f"Render shape: {render_img.shape}")

# Resize render → match right
if render_img.shape[:2] != right_img.shape[:2]:
    print("Resizing rendered image...")
    render_pil = Image.fromarray(render_img)
    render_resized = render_pil.resize(
        (right_img.shape[1], right_img.shape[0]),
        Image.Resampling.LANCZOS
    )
    render_img = np.array(render_resized)
    print(f"After resize: {render_img.shape}")

assert right_img.shape == render_img.shape

# ────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────
psnr_value = compute_psnr(right_img, render_img)
ssim_value = compute_ssim(right_img, render_img)

print("\n" + "═"*60)
print(f"  PSNR : {psnr_value:6.2f} dB")
print(f"  SSIM : {ssim_value:.4f}")
print("═"*60)

# ────────────────────────────────────────────────
#  Error calculation
# ────────────────────────────────────────────────
abs_error = np.abs(right_img.astype(np.float32) - render_img.astype(np.float32))
abs_error_rgb_mean = abs_error.mean(axis=2)          # average over channels

abs_error_mean = abs_error_rgb_mean.mean()
abs_error_max  = abs_error_rgb_mean.max()

print(f"Mean absolute error  : {abs_error_mean:.3f}  (max: {abs_error_max:.1f})")

# ────────────────────────────────────────────────
#  Create error heatmap + colorbar
# ────────────────────────────────────────────────
# Robust normalization (ignore top 2% outliers)
p98 = np.percentile(abs_error_rgb_mean, 98)
error_norm = abs_error_rgb_mean / (p98 + 1e-8)
error_norm = np.clip(error_norm, 0, 1.0)

# ──── Plot with colorbar ────
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(error_norm, cmap='jet', interpolation='nearest')
ax.set_title(f'Error Heatmap – {name}  (red = large difference)')
ax.axis('off')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Normalized absolute error\n(0 = perfect match, 1 = large difference)', rotation=90, labelpad=15)

# Save version with colorbar
base = os.path.join(output_dir, f"eval_{name}")
plt.savefig(f"{base}_error_heatmap_with_colorbar.png", bbox_inches='tight', dpi=150)
plt.close(fig)

print(f"Saved error heatmap with colorbar:\n  {base}_error_heatmap_with_colorbar.png")

# ──── Also save plain version (no colorbar) if you still want it ────
plt.imsave(
    f"{base}_error_heatmap_jet_plain.png",
    error_norm,
    cmap='jet',
    vmin=0, vmax=1
)
print(f"Plain heatmap (no colorbar):\n  {base}_error_heatmap_jet_plain.png")

# ────────────────────────────────────────────────
#  Other saved files (for completeness)
# ────────────────────────────────────────────────
Image.fromarray(right_img).save(f"{base}_right.png")
Image.fromarray(render_img).save(f"{base}_render.png")

side = np.hstack([right_img, render_img])
Image.fromarray(side).save(f"{base}_sidebyside.png")

# raw absolute error (grayscale-ish)
Image.fromarray((abs_error_rgb_mean / abs_error_max * 255).clip(0,255).astype(np.uint8)).save(
    f"{base}_abs_error_grayscale.png"
)

print("\nOther files saved:")
print(f"  {base}_sidebyside.png")
print(f"  {base}_abs_error_grayscale.png")