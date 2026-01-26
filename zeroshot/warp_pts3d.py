import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image  #
import os
from PIL import Image

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



if __name__ == "__main__":
    data_dir = "/data_new/luxiaoxi/dataset/medical_depth/eyetube/fundus/test"
    # output_dir = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_real/MASt3R_0124_real"
    output_dir = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_real/MASt3R_0126_zeroshot"

    ##-----load_image----##
    name = "01504"
    right_image = read_image(os.path.join(data_dir, "right", "%s.jpg" % name))  # (C, H, W)
    left_image = read_image(os.path.join(data_dir, "left", "%s.jpg" % name))  # (C, H, W)
    # Transpose to (H, W, C)
    right_image = right_image.permute(1, 2, 0)
    left_image = left_image.permute(1, 2, 0)

    ##-----load pointmap----##
    # pts3d_left = np.load(os.path.join(output_dir, "pts3d_left_%s.npy"%name))  # (H, W, 3)
    pts3d_right = np.load(os.path.join(output_dir, "pts3d_right_%s.npy"%name)) # (H, W, 3)

    ##-----load intrinsic----##
    intrinsic_file = os.path.join(output_dir, "%s_pred_intrinsics.txt"%name)
    lines = []
    with open(intrinsic_file, "r") as f:
        for l in f.readlines():
            lines.append(l.strip().split(" "))
    fx, fy, cx, cy = float(lines[-1][0]), float(lines[-1][4]), float(lines[-1][2]), float(lines[-1][5])
    K_right = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    K_left = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    ##-----load_extrinsic----##
    extrinsic_file = os.path.join(output_dir, "%s_pred_traj.txt"%name)
    lines = []
    with open(extrinsic_file, "r") as f:
        for l in f.readlines():
            lines.append(l.strip().split(" "))
    num, tx, ty, tz, w, x, y, z = [float(p) for p in lines[-1]]
    q_left = torch.tensor([w, x, y, z])  # Quaternion for left camera
    t_left = torch.tensor([tx, ty, tz])  # Translation for left camera
    # t_left[2] = -t_left[2]
    M_right = torch.eye(4, dtype=torch.float32)  # Assume right is reference
    M_left = quaternion_translation_to_matrix(q_left, t_left)

    # ────────────────────────────────────────────────
    #  Pick device (GPU is strongly recommended)
    # ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move everything to torch & device
    pts3d_right = torch.from_numpy(pts3d_right).to(device)  # (H, W, 3)

    K_right = K_right.to(device)
    K_left = K_left.to(device)
    M_left = M_left.to(device)  # left-to-right 4×4
    right_image = right_image.to(device).float() / 255.0  # (H,W,3) ∈ [0,1]

    H, W, _ = pts3d_right.shape

    # ────────────────────────────────────────────────
    #  1. Transform point cloud from RIGHT → LEFT coord
    # ────────────────────────────────────────────────
    # M_left is the pose of left camera in right camera frame
    #    → we want the inverse = right-to-left transformation
    M_right_to_left = torch.inverse(M_left)  # (4,4)
    # M_right_to_left = M_left

    # homogeneous points  (H,W,4)
    ones = torch.ones(H, W, 1, device=device)
    pts_homo = torch.cat([pts3d_right, ones], dim=-1)  # (H,W,4)

    # transform → left camera coordinate frame
    pts_left = torch.einsum("ij,hwj->hwi", M_right_to_left, pts_homo)  # (H,W,4)

    # de-homogenize
    mask_valid = pts_left[..., 3] > 1e-6
    z_left = pts_left[..., 2].clamp(min=1e-6)  # depth in left cam
    pts_left[..., :3] = pts_left[..., :3] / z_left.unsqueeze(-1)

    # # Insert right after computing z_left  (or before transforming points)
    # scale_factor = 0.01  # ← try values: 0.01, 0.03, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0
    # pts_left[..., 2] *= scale_factor
    # pts_left[..., 0] *= scale_factor  # x also needs scaling (very important!)
    # pts_left[..., 1] *= scale_factor  # same for y

    # desired_median_depth = 1.5  # arbitrary — try 0.8 to 3.0
    # current_median = z_left.median()
    # scale = desired_median_depth / current_median
    # print(f"Applying depth scale: {scale:.4f}")
    #
    # pts_left[..., :3] *= scale

    # Then continue with projection as before

    # Debug prints — run and look at the console
    print("z_left min / max / median:",
          z_left.min().item(),
          z_left.max().item(),
          z_left.median().item())

    print("percentage of positive Z:", (z_left > 0).float().mean().item() * 100, "%")

    # ────────────────────────────────────────────────
    #  2. Project to left image plane (pixel coordinates)
    # ────────────────────────────────────────────────
    uv_left = torch.einsum("ij,hwj->hwi", K_left, pts_left[..., :3])  # (H,W,3)
    u = uv_left[..., 0] / uv_left[..., 2]  # (H,W)
    v = uv_left[..., 1] / uv_left[..., 2]

    print("u  min/median/max:", u.min().item(), u.median().item(), u.max().item())
    print("v  min/median/max:", v.min().item(), v.median().item(), v.max().item())

    valid_in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    print(f"Percentage of points projecting inside image: {valid_in_image.float().mean().item() * 100:.2f} %")

    # normalize to [-1,1] for grid_sample
    u_norm = (u - (W - 1) / 2) / ((W - 1) / 2)  # [-1,1]
    v_norm = (v - (H - 1) / 2) / ((H - 1) / 2)
    grid = torch.stack([u_norm, v_norm], dim=-1)  # (H,W,2)



    # ────────────────────────────────────────────────
    #  3. Optional: depth-based occlusion / z-buffer
    #     (simple nearest neighbor splatting approximation)
    # ────────────────────────────────────────────────
    depth_render = torch.zeros(H, W, device=device)
    depth_render = z_left.clone()
    depth_render[~mask_valid] = 1e9  # invalid → far

    # ────────────────────────────────────────────────
    #  4. Warp RGB using grid_sample (main rendering step)
    # ────────────────────────────────────────────────
    # (N=1, C=3, H, W) input format expected by F.grid_sample
    src_img = right_image.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    rendered = F.grid_sample(
        src_img,
        grid.unsqueeze(0),  # (1,H,W,2)
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # (1,3,H,W)

    rendered = rendered.squeeze(0).permute(1, 2, 0)  # (H,W,3)

    # optional: mask out invalid / very far regions
    # valid = (grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1) & mask_valid & (z_left < 5.0)  # ← tune 5.0
    # rendered[~valid] = 0  # or torch.nan, or left_image value, etc.

    # ────────────────────────────────────────────────
    #  Save / visualize
    # ────────────────────────────────────────────────
    out_path = os.path.join(output_dir, f"reproj_left_from_right_{name}.png")
    Image.fromarray((rendered.cpu().numpy() * 255).astype(np.uint8)).save(out_path)

    print("Saved:", out_path)

    # Optional: also save depth for debugging
    depth_vis = (depth_render / depth_render.quantile(0.98).clamp(1e-4)).clamp(0, 1)
    Image.fromarray((depth_vis.cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"depth_reproj_{name}.png")
    )

