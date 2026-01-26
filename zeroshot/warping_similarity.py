import os.path

import torch
import torch.nn.functional as F
from torchvision.io import read_image  # Assuming torchvision is available; otherwise, use PIL or OpenCV to load images


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        q (torch.Tensor): Quaternion tensor of shape (4,) in the order [w, x, y, z].
                          Assumes the quaternion is normalized.

    Returns:
        torch.Tensor: 3x3 rotation matrix.
    """
    # Normalize the quaternion if not already (optional, but assuming it is)
    # q = q / torch.norm(q)

    w, x, y, z = q[0], q[1], q[2], q[3]

    r00 = 1 - 2 * (y ** 2 + z ** 2)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)

    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x ** 2 + z ** 2)
    r12 = 2 * (y * z - x * w)

    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x ** 2 + y ** 2)

    R = torch.tensor([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])

    return R


def quaternion_translation_to_RT(q: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert quaternion and translation to rotation matrix R and translation vector T.

    Args:
        q (torch.Tensor): Quaternion tensor of shape (4,) in [w, x, y, z] order.
        t (torch.Tensor): Translation vector of shape (3,).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (R: 3x3 rotation matrix, T: 3x1 translation vector)
    """
    R = quaternion_to_rotation_matrix(q)
    T = t.view(3, 1)  # Ensure it's a column vector
    return R, T


def quaternion_translation_to_matrix(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion and translation to a 4x4 transformation matrix M.

    Args:
        q (torch.Tensor): Quaternion tensor of shape (4,) in [w, x, y, z] order.
        t (torch.Tensor): Translation vector of shape (3,).

    Returns:
        torch.Tensor: 4x4 transformation matrix.
    """
    R = quaternion_to_rotation_matrix(q)
    M = torch.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def get_uv_coords(height: int,
                  width: int,
                  is_homo: bool = False) -> torch.Tensor:
    """
    Helper function that returns the uv coordinates of the image.
    """
    x_, y_ = torch.meshgrid(torch.arange(width),
                            torch.arange(height),
                            indexing='xy')
    uv_coords = torch.dstack([x_, y_])  # (H, W, 2)

    if is_homo:  # Add 1s to make homogeneous
        ones = torch.ones_like(x_)  # (H, W)
        uv_coords = torch.dstack([uv_coords, ones])  # (H, W, 3)

    return uv_coords.float()


def get_camera_transformation(depthmap: torch.Tensor,
                              K1: torch.Tensor,
                              K2: torch.Tensor,
                              M1: torch.Tensor,
                              M2: torch.Tensor) -> torch.Tensor:
    """
    This function returns the camera transformation matrix between two given camera matrices and a depthmap.
    """
    height, width = depthmap.shape

    # Get the transformation matrix from first viewpoint to the second view point.
    T21 = M2 @ torch.linalg.inv(M1)  # (4, 4)

    # Create homogeneous image coordinates for the camera viewpoint
    uv_coords = get_uv_coords(height, width, is_homo=True)  # (H, W, 3)

    # Un-project the image coordinates to the world coordinates
    K1_inv = torch.linalg.inv(K1)[None, None, ...]  # (1, 1, 3, 3)
    world_coords = K1_inv @ uv_coords[..., None]  # (H, W, 3, 1)

    # scale the world coordinates by the depth
    world_coords = depthmap[..., None, None] * world_coords  # (H, W, 3, 1)

    # Transform the world coordinates based on the transformation between the two cameras.
    # but before that, make the world coords homogeneous
    ones = torch.ones(height, width)[..., None, None]
    world_coords = torch.cat([world_coords, ones], dim=-2)  # (H, W, 4, 1)
    world_coords_trans = T21[None, None] @ world_coords  # (H, W, 4, 1)

    world_coords_trans = world_coords_trans[..., :3, :]  # (H, W, 3, 1)
    # Reproject the transformed world coordinates to the 2nd camera viewpoint
    trans_coords = K2[None, None, ...] @ world_coords_trans  # (H, W, 3, 1)

    return trans_coords[..., 0]  # ignore the last dim in the homogeneous form (H, W, 3)


def splat_points(image: torch.Tensor,
                 weights: torch.Tensor,
                 flowmap: torch.Tensor,
                 do_exp_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Splat the image pixels onto the transformed positions as dictated by the flowmap and the depth weights.
    """
    if not torch.is_floating_point(image):
        image = image.float()
    height, width, channels = image.shape

    # Step 1: Get the transformed positions
    uv_coords = get_uv_coords(height, width)  # (H, W, 2)
    trans_coords = flowmap + uv_coords  # (H, W, 2)

    trans_coords = trans_coords + 0.5  # Adjust for subpixel accuracy if needed

    trans_coords_low = torch.floor(trans_coords).long()  # (H, W, 2)
    trans_coords_high = torch.ceil(trans_coords).long()  # (H, W, 2)

    # Clamp the transformed coordinates to the image boundaries
    trans_coords_low[..., 0].clamp_(min=0, max=width - 1)
    trans_coords_low[..., 1].clamp_(min=0, max=height - 1)
    trans_coords_high[..., 0].clamp_(min=0, max=width)
    trans_coords_high[..., 1].clamp_(min=0, max=height)

    # Calculate the weights for the four neighboring pixels (Inverse bilinear interpolation)
    dx = trans_coords[..., 0:1] - trans_coords_low[..., 0:1].float()
    dy = trans_coords[..., 1:2] - trans_coords_low[..., 1:2].float()

    flow_nw = (1 - dx) * (1 - dy)
    flow_ne = dx * (1 - dy)
    flow_sw = (1 - dx) * dy
    flow_se = dx * dy

    # Augment the flow offsets based on the depth weights
    if do_exp_weights:
        exp_weights = torch.exp(1.0 / weights[..., None])
        flow_nw = flow_nw * exp_weights
        flow_ne = flow_ne * exp_weights
        flow_sw = flow_sw * exp_weights
        flow_se = flow_se * exp_weights
    else:
        inv_weights = 1.0 / weights[..., None]
        flow_nw = flow_nw * inv_weights
        flow_ne = flow_ne * inv_weights
        flow_sw = flow_sw * inv_weights
        flow_se = flow_se * inv_weights

    # Prepare expanded image and weights for scattering (pad to avoid index errors)
    padded_height, padded_width = height + 2, width + 2
    splat_image = torch.zeros((padded_height, padded_width, channels), device=image.device, dtype=image.dtype)
    weights_acc = torch.zeros((padded_height, padded_width, 1), device=image.device, dtype=image.dtype)

    # Adjust indices for padding (add 1)
    low_y = trans_coords_low[..., 1] + 1
    low_x = trans_coords_low[..., 0] + 1
    high_y = trans_coords_high[..., 1] + 1
    high_x = trans_coords_high[..., 0] + 1

    # Flatten everything
    indices_nw = (low_y * padded_width + low_x).flatten()
    indices_ne = (low_y * padded_width + high_x).flatten()
    indices_sw = (high_y * padded_width + low_x).flatten()
    indices_se = (high_y * padded_width + high_x).flatten()

    flat_splat = splat_image.view(-1, channels)
    flat_weights = weights_acc.view(-1, 1)

    flat_splat.scatter_add_(0, indices_nw[:, None].expand(-1, channels),
                            (image * flow_nw[..., 0, None]).view(-1, channels))
    flat_splat.scatter_add_(0, indices_ne[:, None].expand(-1, channels),
                            (image * flow_ne[..., 0, None]).view(-1, channels))
    flat_splat.scatter_add_(0, indices_sw[:, None].expand(-1, channels),
                            (image * flow_sw[..., 0, None]).view(-1, channels))
    flat_splat.scatter_add_(0, indices_se[:, None].expand(-1, channels),
                            (image * flow_se[..., 0, None]).view(-1, channels))

    flat_weights.scatter_add_(0, indices_nw[:, None], flow_nw[..., 0, None].view(-1, 1))
    flat_weights.scatter_add_(0, indices_ne[:, None], flow_ne[..., 0, None].view(-1, 1))
    flat_weights.scatter_add_(0, indices_sw[:, None], flow_sw[..., 0, None].view(-1, 1))
    flat_weights.scatter_add_(0, indices_se[:, None], flow_se[..., 0, None].view(-1, 1))

    splat_image = flat_splat.view(padded_height, padded_width, channels)
    weights_acc = flat_weights.view(padded_height, padded_width, 1)

    # Center crop
    cropped_splat = splat_image[1:-1, 1:-1, :]
    cropped_weights = weights_acc[1:-1, 1:-1, :]

    mask = cropped_weights > 1e-6

    cropped_splat = cropped_splat / cropped_weights.clamp(min=1e-6)

    cropped_splat[~mask.repeat(1, 1, channels)] = 0

    return cropped_splat, mask[..., 0]


def get_weights_from_depthmap(depthmap: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """
    Compute weights from depth map. Larger gamma emphasizes closer points more strongly.
    """
    return depthmap.clamp(min=1e-3).pow(gamma)


def forward_warping(
        image: torch.Tensor,  # (H, W, 3) uint8 or float
        depthmap: torch.Tensor,  # (H, W) float, depth from right camera
        K1: torch.Tensor,  # (3,3) intrinsic matrix - source camera
        M1: torch.Tensor,  # (4,4) pose matrix - source camera [R|t]
        M2: torch.Tensor,  # (4,4) pose matrix - target camera [R|t]
        K2: torch.Tensor = None,  # (3,3) intrinsic - target (optional, same as K1 by default)
        gamma: float = 5.0,
        do_exp_weights: bool = False,
        eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward warping using direct 3D→2D projection instead of flow field.

    Args:
        image:      source image (right) - (H, W, C)
        depthmap:   depth map in source camera coordinates (right)
        K1, M1:     intrinsics & extrinsics of source camera
        M2:         extrinsics of target camera (where we want to render)
        K2:         intrinsics of target camera (if different)

    Returns:
        warped_image:   rendered image in target view (H, W, C)
        valid_mask:     splatting coverage mask (H, W)
    """
    if K2 is None:
        K2 = K1.clone()

    H, W, C = image.shape
    device = image.device

    # 1. Create pixel grid in source image (homogeneous)
    uv = get_uv_coords(H, W, is_homo=True)  # (H, W, 3)
    uv = uv.to(device)

    # 2. Back-project to 3D points in source camera coordinate
    K1_inv = torch.linalg.inv(K1)
    pts_cam1 = (K1_inv @ uv[..., None]).squeeze(-1)  # (H,W,3)
    pts_cam1 = pts_cam1 * depthmap[..., None]  # scale by depth  → (H,W,3)

    # 3. Transform to world coordinate
    # M1 = [R1 | t1; 0 0 0 1]  →  world = R1^T * (cam1 - t1)  or depending on your convention
    # Most common OpenCV-like convention: cam = R @ world + t  →  world = R^T @ (cam - t)
    # Here we assume M = [R | t; 0 0 0 1] where cam_coord = R @ world + t

    M1_inv = torch.linalg.inv(M1)
    pts_world_hom = M1_inv @ torch.cat([pts_cam1, torch.ones(H, W, 1, device=device)], dim=-1)[..., None]
    pts_world = pts_world_hom[..., :3, 0]  # (H,W,3)

    # 4. Transform to target camera coordinate
    pts_cam2_hom = M2 @ torch.cat([pts_world, torch.ones(H, W, 1, device=device)], dim=-1)[..., None]
    pts_cam2 = pts_cam2_hom[..., :3, 0]  # (H,W,3)

    # 5. Project to target image plane
    proj_hom = K2 @ pts_cam2[..., None]  # (H,W,3,1)
    proj_hom = proj_hom.squeeze(-1)  # (H,W,3)

    # Get projected pixel coordinates + depth
    z = proj_hom[..., 2].clamp(min=eps)  # avoid division by zero or negative
    uv_proj = proj_hom[..., :2] / z[..., None]  # (H,W,2)

    # ────────────────────────────────────────────────
    # Now splat using the projected coordinates
    # ────────────────────────────────────────────────

    # For stability, we can also use the target depth as weight (optional)
    target_depth = z.clamp(min=1e-3)

    # Prepare weights (closer = higher weight)
    if do_exp_weights:
        weights = torch.exp(gamma / target_depth)
    else:
        weights = target_depth.pow(-gamma)  # 1 / depth^gamma

    # ── Splat ────────────────────────────────────────
    warped, valid = splat_points(
        image=image.float(),
        weights=weights,
        flowmap=uv_proj,  # ← here we directly use projected coords as target position
        do_exp_weights=False  # already applied in weights
    )

    # Final touch-up
    warped = warped.clamp(0, 255).round().to(torch.uint8)

    return warped, valid

def gaussian_window(size, sigma):
    kernel = torch.zeros(size, dtype=torch.float32)
    center = size // 2
    denom = 2.0 * sigma * sigma

    for x in range(size):
        diff = float(x - center)
        val = torch.exp(torch.tensor(-diff * diff / denom))
        kernel[x] = val

    return kernel / kernel.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian_window(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, sigma=1.5):
    (_, h, w, channel) = img1.shape
    window = create_window(window_size, sigma, channel).to(img1.device)

    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def psnr(img1, img2):
    img1 = img1.float()
    img2 = img2.float()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

# Example usage
import numpy as np
import torchvision
# Load images (assume RGB, uint8, and depth is float single channel)
data_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/part4"
output_dir = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_real/MASt3R_0124_real"

name = "01351"
# Define intrinsics and extrinsics (example values; replace with your own)
fx, fy, cx, cy = 2027.853638, 2027.853638,  33.062080, 251.143936
# fx, fy, cx, cy = 2027.853638, 2027.853638,  320, 180
K_right = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
K_left = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
w, x, y, z = 0.657205, -0.244486, 0.310203, 0.641936
tx, ty, tz = -7.959595, -43.012802, -32.671700

right_image = read_image(os.path.join(data_dir, "right", "%s.jpg"%name))  # (C, H, W)
left_image = read_image(os.path.join(data_dir, "left", "%s.jpg"%name))  # (C, H, W)

# depth_right_o = torch.from_numpy(np.load(os.path.join(output_dir, "preddepth_%s.npy"%name))).float()
# depth_right_o = torch.from_numpy(np.load(os.path.join(output_dir, "preddepth_left_%s.npy"%name))).float()
depth_right_o = torch.from_numpy(np.load(os.path.join(output_dir, "01351_depthmaps.txt/metric_depth/right/000000.npy"))).float()[0]

resize = torchvision.transforms.Resize((360, 640))
depth_right = resize(depth_right_o.unsqueeze(0)).squeeze(0)
print(depth_right.shape)
# depth_right = read_image(os.path.join(output_dir, )).float()[0]  # Take first channel, (H, W)

# depth_right = read_image(os.path.join(output_dir, "depth_%s.png"%name)).float()[0]  # Take first channel, (H, W)




# name = "03422"
# # Define intrinsics and extrinsics (example values; replace with your own)
# fx, fy, cx, cy = 2939.470703, 2939.470703,  261.008423, 75.062012
# K_right = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
# K_left = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
# w, x, y, z = 0.7603292593334942, 0.3372800835200532, 0.4447126475585373, -0.33222315356332766
# tx, ty, tz = -37.248687744140625, 62.45252227783203, -30.208410263061523
#
# right_image = read_image(os.path.join(data_dir, "right", "%s.jpg"%name))  # (C, H, W)
# depth_right_o = torch.from_numpy(np.load(os.path.join(output_dir, "preddepth_%s.npy"%name))).float()
# # depth_right_o = torch.from_numpy(np.load(os.path.join(output_dir, "preddepth_left_%s.npy"%name))).float()
#
# resize = torchvision.transforms.Resize((360, 640))
# depth_right = resize(depth_right_o.unsqueeze(0)).squeeze(0)
# # depth_right = read_image(os.path.join(output_dir, )).float()[0]  # Take first channel, (H, W)
# left_image = read_image(os.path.join(data_dir, "left", "%s.jpg"%name))  # (C, H, W)

# name = "06732"
# # Define intrinsics and extrinsics (example values; replace with your own)
# fx, fy, cx, cy = 3243.594482, 3243.594482,  267.292511, 105.977692
# K_right = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
# K_left = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
# w, x, y, z = 0.8165736984331317, -0.20312652363536504, 0.04955960428229395, 0.5380435447493361
# tx, ty, tz = 13.096526145935059, -33.47565841674805, -78.916259765625
#
# right_image = read_image(os.path.join(data_dir, "right", "%s.jpg"%name))  # (C, H, W)
# depth_right_o = torch.from_numpy(np.load(os.path.join(output_dir, "preddepth_%s.npy"%name))).float()
# # depth_right_o = torch.from_numpy(np.load(os.path.join(output_dir, "preddepth_left_%s.npy"%name))).float()
#
# resize = torchvision.transforms.Resize((540, 960))
# depth_right = resize(depth_right_o.unsqueeze(0)).squeeze(0)
# # depth_right = read_image(os.path.join(output_dir, )).float()[0]  # Take first channel, (H, W)
# left_image = read_image(os.path.join(data_dir, "left", "%s.jpg"%name))  # (C, H, W)



# Transpose to (H, W, C)
right_image = right_image.permute(1, 2, 0)
left_image = left_image.permute(1, 2, 0)



q_left = torch.tensor([w, x, y, z])  # Quaternion for left camera
t_left = torch.tensor([tx, ty, tz])  # Translation for left camera
# t_left[0] = -t_left[0]
M_right = torch.eye(4, dtype=torch.float32)  # Assume right is reference
M_left = quaternion_translation_to_matrix(q_left, t_left).inverse()

# Render left from right + depth
rendered_left, mask = forward_warping(image=right_image,
                                      depthmap=depth_right,
                                      K1=K_right,
                                      M1=M_right,
                                      M2=M_left,
                                      K2=K_left)

# Difference (absolute error)
error = torch.abs(rendered_left.float() - left_image.float()).mean(dim=-1)  # (H,W)

rendered_np = rendered_left.cpu().numpy()

# Create PIL Image
from PIL import Image
pil_img = Image.fromarray(rendered_np)

# Save to file
pil_img.save(os.path.join(output_dir, "rendered_left.png"))

# To compute metrics on valid pixels
valid_rendered = rendered_left[mask]
valid_original = left_image[mask]

# But for SSIM/PSNR, since they are global, compute on whole or masked
# Here, compute on whole for simplicity

ssim_value = ssim(rendered_left.unsqueeze(0).float(), left_image.unsqueeze(0).float())
psnr_value = psnr(rendered_left, left_image)

print(f"SSIM: {ssim_value.item()}")
print(f"PSNR: {psnr_value.item()}")

# Note: If mask is needed for accurate metrics, mask out holes in both images or inpaint, but this is basic.
# Also, ensure depth units match camera params (e.g., meters if T is in meters).
# The splat_points function uses scatter_add_ for efficiency.
# If errors, adjust clamping or padding.