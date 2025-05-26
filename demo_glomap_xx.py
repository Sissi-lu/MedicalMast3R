#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Script to perform SfM with MASt3R and GloMap, outputting camera poses, depth maps, and FPS metrics.
#
import os
import argparse
import numpy as np
import torch
import pycolmap
import PIL.Image
import tempfile
import shutil
import time
import matplotlib.cm as cm
from pathlib import Path
from scipy.spatial.transform import Rotation

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.mapping import kapture_import_image_folder_or_list, run_mast3r_matching, glomap_run_mapper
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
# from dust3r.viz import depth_to_rgb
from kapture.converter.colmap.database_extra import kapture_to_colmap


def get_args_parser():
    parser = argparse.ArgumentParser(description="MASt3R SfM with pose, depth map, and FPS output")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save poses, depth maps, and metrics")
    parser.add_argument('--weights', type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                        help="Path to MASt3R weights or model name")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to run the model on")
    parser.add_argument('--image_size', type=int, default=512, help="Size to resize images")
    parser.add_argument('--glomap_bin', type=str, default='glomap', help="Path to GloMap binary")
    parser.add_argument('--scenegraph_type', type=str, default='complete',
                        choices=['complete', 'swin', 'logwin', 'oneref'], help="Scene graph type for pairing")
    parser.add_argument('--winsize', type=int, default=1, help="Window size for swin/logwin scene graph")
    parser.add_argument('--shared_intrinsics', action='store_true', help="Optimize one set of intrinsics for all views")
    parser.add_argument('--save_pointcloud', action='store_true', help="Save sparse point cloud as .ply")
    return parser

def save_poses(world_to_cam, image_names, output_dir):
    """Save camera poses as a text file and NumPy array."""
    os.makedirs(output_dir, exist_ok=True)
    pose_file = os.path.join(output_dir, 'poses.txt')
    with open(pose_file, 'w') as f:
        f.write("# Image_name tx ty tz qx qy qz qw\n")
        for img_id, pose_w2c in world_to_cam.items():
            img_name = image_names[img_id]
            t = pose_w2c[:3, 3]  # Translation
            R = pose_w2c[:3, :3]  # Rotation matrix
            quat = Rotation.from_matrix(R).as_quat()  # Quaternion (x, y, z, w)






    print(f"Poses saved to {pose_file} and poses.npy")


def save_depth_maps(image_names, world_to_cam, intrinsics, points3d, images, output_dir, metrics, image_size, interpolate=False):
    """Generate and save depth maps by projecting 3D point cloud onto images using camera poses."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    num_images = len(images)

    for img_id, img_array in images.items():
        img_name = image_names[img_id]
        # Get camera pose (world-to-camera) and intrinsics
        pose_w2c = world_to_cam[img_id]  # 4x4 matrix
        K = intrinsics[img_id]  # 3x3 intrinsics matrix

        # Initialize depth map
        h, w = img_array.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)

        # Project 3D points to 2D image plane
        points = np.array([p[0] for p in points3d], dtype=np.float32)  # Nx3
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))  # Nx4 (homogeneous)
        cam_points = (pose_w2c @ points_hom.T).T  # Transform to camera coordinates
        depths = cam_points[:, 2]  # Z-coordinate in camera frame (depth)

        # Project to image plane
        valid_mask = depths > 0  # Only project points in front of the camera
        cam_points = cam_points[valid_mask]
        depths = depths[valid_mask]
        if len(depths) == 0:
            print(f"Warning: No valid 3D points project to {img_name}. Skipping depth map.")
            continue

        # Apply camera intrinsics
        img_points = (K @ cam_points[:, :3].T).T  # Nx3 (x, y, z)
        img_points_2d = img_points[:, :2] / img_points[:, 2:3]  # Nx2 (u, v)

        # Filter points within image bounds
        u, v = img_points_2d[:, 0], img_points_2d[:, 1]
        valid_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[valid_bounds].astype(int), v[valid_bounds].astype(int)
        depths = depths[valid_bounds]

        # Assign depth values to pixels
        depth_map[v, u] = depths

        # Optional: Interpolate sparse depth map
        if interpolate:
            from scipy.ndimage import map_coordinates
            valid_pixels = depth_map > 0
            if np.any(valid_pixels):
                coords = np.where(valid_pixels)
                values = depth_map[coords]
                grid_x, grid_y = np.mgrid[0:h, 0:w]
                depth_map = map_coordinates(values, [coords[0], coords[1]], output=depth_map, order=1, mode='constant',
                                            cval=0.0)

        # Save depth map as .npy
        depth_file = os.path.join(output_dir, f"depth_{img_name.replace('/', '_')}.npy")
        np.save(depth_file, depth_map)

        # Save depth map as RGB visualization
        depth_rgb = depth_to_rgb(depth_map)
        depth_img = PIL.Image.fromarray(depth_rgb)
        depth_img.save(os.path.join(output_dir, f"depth_{img_name.replace('/', '_')}.png"))
        print(f"Depth map for {img_name} saved to {depth_file} and .png")

    elapsed = time.time() - start_time
    fps = num_images / elapsed if elapsed > 0 else float('inf')
    metrics['depth_maps'] = {'time': elapsed, 'num_images': num_images, 'fps': fps}
    print(f"Depth map generation: {num_images} images in {elapsed:.2f}s, FPS: {fps:.2f}")






def depth_to_rgb(depth_map):
    """
    Convert a depth map to an RGB visualization using a colormap.

    Args:
        depth_map (np.ndarray): 2D array of depth values (float32).

    Returns:
        np.ndarray: RGB image (H, W, 3) with uint8 values.
    """
    # Create a copy to avoid modifying the input
    depth_map = depth_map.copy()

    # Mask out zero values (common in sparse depth maps)
    valid_mask = depth_map > 0
    if not np.any(valid_mask):
        # If no valid depths, return a black image
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    # Normalize depth values to [0, 1] based on valid depths
    valid_depths = depth_map[valid_mask]
    depth_min, depth_max = valid_depths.min(), valid_depths.max()
    if depth_max == depth_min:
        # Avoid division by zero if all depths are the same
        normalized_depth = np.zeros_like(depth_map)
        normalized_depth[valid_mask] = 1.0
    else:
        normalized_depth = np.zeros_like(depth_map)
        normalized_depth[valid_mask] = (depth_map[valid_mask] - depth_min) / (depth_max - depth_min)

    # Apply colormap (jet is a common choice for depth visualization)
    cmap = cm.get_cmap('jet')
    rgb = cmap(normalized_depth)[:, :, :3]  # Take RGB channels, discard alpha
    rgb = (rgb * 255).astype(np.uint8)  # Convert to uint8
    return rgb

def save_pointcloud(points3d, output_dir):
    """Save sparse 3D point cloud as a .ply file."""
    from plyfile import PlyData, PlyElement
    points = np.array([p[0] for p in points3d], dtype=np.float32)
    colors = np.array([p[1] for p in points3d], dtype=np.uint8)
    vertex = np.array([(p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points, colors)],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    ply_file = os.path.join(output_dir, 'pointcloud.ply')
    PlyData([el]).write(ply_file)
    print(f"Point cloud saved to {ply_file}")

def save_metrics(metrics, output_dir):
    """Save performance metrics to a text file."""
    metrics_file = os.path.join(output_dir, 'performance_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Performance Metrics\n")
        f.write("=================\n")
        for stage, data in metrics.items():
            f.write(f"{stage}:\n")
            f.write(f"  Time: {data['time']:.2f} seconds\n")
            if 'num_images' in data:
                f.write(f"  Images: {data['num_images']}\n")
                f.write(f"  FPS: {data['fps']:.2f}\n")
            elif 'num_pairs' in data:
                f.write(f"  Pairs: {data['num_pairs']}\n")
                f.write(f"  FPS: {data['fps']:.2f}\n")
        total_time = sum(data['time'] for data in metrics.values())
        total_images = metrics.get('image_loading', {}).get('num_images', 0)
        overall_fps = total_images / total_time if total_time > 0 and total_images > 0 else float('inf')
        f.write("Overall:\n")
        f.write(f"  Total Time: {total_time:.2f} seconds\n")
        f.write(f"  Total Images: {total_images}\n")
        f.write(f"  Overall FPS: {overall_fps:.2f}\n")
    print(f"Metrics saved to {metrics_file}")

def main():
    args = get_args_parser().parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    metrics = {}  # Store timing and FPS for each stage

    # Start total pipeline timer
    total_start_time = time.time()

    # Load MASt3R model
    model = AsymmetricMASt3R.from_pretrained(args.weights).to(args.device)
    print(f"Loaded MASt3R model on {args.device}")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load images from input directory
    start_time = time.time()
    input_dir = args.input_dir
    filelist = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
    if not filelist:
        raise ValueError(f"No images found in {input_dir}")
    imgs = load_images(filelist, size=args.image_size)
    elapsed = time.time() - start_time
    num_images = len(imgs)
    fps = num_images / elapsed if elapsed > 0 else float('inf')
    metrics['image_loading'] = {'time': elapsed, 'num_images': num_images, 'fps': fps}
    print(f"Image loading: {num_images} images in {elapsed:.2f}s, FPS: {fps:.2f}")

    # Create scene graph for image pairs
    start_time = time.time()
    scene_graph = args.scenegraph_type
    if scene_graph in ['swin', 'logwin']:
        scene_graph = f"{scene_graph}-{args.winsize}"
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    elapsed = time.time() - start_time
    metrics['pair_creation'] = {'time': elapsed}
    print(f"Pair creation: {len(pairs)} pairs in {elapsed:.2f}s")

    # Setup temporary directory for COLMAP and GloMap
    with tempfile.TemporaryDirectory(suffix='_mast3r_sfm') as tmp_dir:
        cache_dir = os.path.join(tmp_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Prepare COLMAP database
        start_time = time.time()
        root_path = os.path.commonpath(filelist)
        filelist_relpath = [os.path.relpath(f, root_path).replace('\\', '/') for f in filelist]
        kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), args.shared_intrinsics)
        image_pairs = [(filelist_relpath[img1['idx']], filelist_relpath[img2['idx']]) for img1, img2 in pairs]
        colmap_db_path = os.path.join(cache_dir, 'colmap.db')
        if os.path.isfile(colmap_db_path):
            os.remove(colmap_db_path)
        os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)

        # Run MASt3R matching
        from kapture.converter.colmap.database import COLMAPDatabase
        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        try:
            kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db)
            colmap_image_pairs = run_mast3r_matching(model, args.image_size, 16, args.device,
                                                     kdata, root_path, image_pairs, colmap_db,
                                                     False, 5, 1.001, False, 3)
            colmap_db.close()
        except Exception as e:
            colmap_db.close()
            raise RuntimeError(f"Matching failed: {e}")
        elapsed = time.time() - start_time
        num_pairs = len(colmap_image_pairs)
        fps = num_pairs / elapsed if elapsed > 0 else float('inf')
        metrics['matching'] = {'time': elapsed, 'num_pairs': num_pairs, 'fps': fps}
        print(f"MASt3R matching: {num_pairs} pairs in {elapsed:.2f}s, FPS: {fps:.2f}")

        if not colmap_image_pairs:
            raise RuntimeError("No matches were kept")

        # Run GloMap for SfM
        start_time = time.time()
        with open(os.path.join(cache_dir, 'pairs.txt'), 'w') as f:
            for img1, img2 in colmap_image_pairs:
                f.write(f"{img1} {img2}\n")
        pycolmap.verify_matches(colmap_db_path, os.path.join(cache_dir, 'pairs.txt'))
        reconstruction_path = os.path.join(cache_dir, 'reconstruction')
        if os.path.isdir(reconstruction_path):
            shutil.rmtree(reconstruction_path)
        os.makedirs(reconstruction_path, exist_ok=True)
        glomap_run_mapper(args.glomap_bin, colmap_db_path, reconstruction_path, root_path)
        elapsed = time.time() - start_time
        fps = num_images / elapsed if elapsed > 0 else float('inf')
        metrics['sfm'] = {'time': elapsed, 'num_images': num_images, 'fps': fps}
        print(f"GloMap SfM: {num_images} images in {elapsed:.2f}s, FPS: {fps:.2f}")

        # Load reconstruction
        recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, '0'))
        print(recon.summary())

        # Extract poses and intrinsics
        world_to_cam = {}
        intrinsics = {}
        image_names = {}
        images = {}
        for colmap_imgid, colmap_image in recon.images.items():
            image_names[colmap_imgid] = colmap_image.name
            world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix()
            camera = recon.cameras[colmap_image.camera_id]
            K = np.eye(3)
            K[0, 0] = camera.focal_length_x
            K[1, 1] = camera.focal_length_y
            K[0, 2] = camera.principal_point_x
            K[1, 2] = camera.principal_point_y
            intrinsics[colmap_imgid] = K
            with PIL.Image.open(os.path.join(root_path, colmap_image.name)) as im:
                images[colmap_imgid] = np.asarray(im)

        # Extract 3D points
        points3d = [(pt3d.xyz, pt3d.color) for pt3d_id, pt3d in recon.points3D.items()]

        # Save outputs
        save_poses(world_to_cam, image_names, os.path.join(output_dir, 'poses'))
        save_depth_maps(image_names, world_to_cam, intrinsics, points3d, images, os.path.join(output_dir, 'depth_maps'), metrics,
                        args.image_size, interpolate=False)
        if args.save_pointcloud:
            save_pointcloud(points3d, output_dir)

    # Save performance metrics
    metrics['total'] = {'time': time.time() - total_start_time, 'num_images': num_images,
                        'fps': num_images / (time.time() - total_start_time) if (time.time() - total_start_time) > 0 else float('inf')}
    save_metrics(metrics, output_dir)
    print(f"Total pipeline: {num_images} images in {metrics['total']['time']:.2f}s, FPS: {metrics['total']['fps']:.2f}")

if __name__ == '__main__':
    main()