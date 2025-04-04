#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
import gradio
import os
import numpy as np
import functools
import sys
sys.path.append('..')
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import argparse
from contextlib import nullcontext
from PIL import Image
from matplotlib import pyplot as plt
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
import torch
import mast3r.utils.path_to_dust3r  # noqa

from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from mast3r.demo import _convert_scene_output_to_glb, main_demo


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    # parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--weights", type=str, help="path of this model", default="./naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--model_name", type=str, help="name of the model weights",
                                default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_temp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False,
                        help='Use ground truth masks for DAVIS')

    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')

    parser.prog = 'mast3r demo'
    return parser

def get_reconstructed_scene(args, img_pair_name, outdir, model, device, silent, image_size, current_scene_state,
                            filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    save_folder = os.path.join(args.output_dir, args.seq_name, img_pair_name)
    os.makedirs(save_folder, exist_ok=True)


    # -----need to modify how to load images-----#
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        # outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)
        outfile_name = os.path.join(save_folder, "scene.glb")
    scene_state = SparseGAState(scene, False, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(save_folder, silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)

    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    # scene.pts3d = pts3d
    scene.depthmaps = depthmaps
    # scene.confs = confs

    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    relative_depth_maps = scene.save_relative_depth_maps(save_folder)

    confidence_masks = to_numpy([c > min_conf_thr for c in confs])

    # dynamic_masks = scene.save_dynamic_masks(save_folder)
    # conf = scene.save_conf_maps(save_folder)
    # init_conf = scene.save_init_conf_maps(save_folder)
    # rgbs = scene.save_rgb_imgs(save_folder)

    # ----------------find 2D-2D matches between the two images------------#


    pts2d_list, pts3d_list = [], []
    imgs = scene.imgs
    for i in range(2):
        conf_i = confidence_masks[i]
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].reshape(imgs[i].shape).detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches

    n_viz = 20
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    plt.savefig(os.path.join(save_folder, "viz_matches.png"))
    plt.close()


    return scene_state, outfile

def get_3D_model_from_scene(save_folder, silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

if __name__ == '__main__':
    parse = get_args_parser()
    args = parse.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

        if args.server_name is not None:
            server_name = args.server_name
        else:
            server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

        if args.weights is not None:
            weights_path = args.weights
        else:
            weights_path = "naver/" + args.model_name

        model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
        chkpt_tag = hash_md5(weights_path)

        # Use the provided output_dir or create a temporary directory
        def get_context(tmp_dir):
            return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
                else nullcontext(tmp_dir)

        if args.output_dir is not None:
            tmpdirname = args.output_dir
        else:
            with get_context(args.tmp_dir) as tmpdirname:
                cache_path = os.path.join(tmpdirname, chkpt_tag)
                os.makedirs(cache_path, exist_ok=True)
            tempdirname = cache_path

        # get in mast3r\demo function def main_demo

        if not args.silent:
            print('Outputing stuff in', tmpdirname)

        if args.input_dir is not None:
            # the same as dust3R, now is the mast3R
            ###-----need to modify the load of images!----------###
            input_files = [[os.path.join(args.input_dir, 'left', fname), os.path.join(args.input_dir, 'right', fname)] for fname in sorted(os.listdir(os.path.join(args.input_dir, "left")))]
            for input_file in input_files:
                file_name = input_file[0].split('/')[-1].split('.')[0]
                recon_fun = functools.partial(get_reconstructed_scene, args, file_name, tmpdirname, model, args.device,
                                              args.silent, args.image_size)  # get_reconstructed_scene有修改
                scene, outfile = recon_fun(
                    current_scene_state=None,
                    filelist=input_file,
                    optim_level="refine+depth",  # choice=["coarse", "refine", "refine+depth"]
                    lr1=0.07,  # Coarese minimum 0.01-0.2
                    niter1=500,  # num_iterations
                    lr2=0.014,  # Fine LR:0.005-0.05
                    niter2=500, # num_iterations
                    min_conf_thr=1.5,  # adjust the confidence threshold0.0-10
                    matching_conf_thr=5., # Matching confidence Thr
                    as_pointcloud=True,
                    mask_sky=False,
                    clean_depth=True,
                    transparent_cams=False,
                    cam_size=0.2,  # adjust the camera size in the output point cloud: 0.001-1
                    scenegraph_type="complete",
                    #[("complete: all possible image pairs", "complete"),
                    # ("swin: sliding window", "swin"),
                    # ("logwin: sliding window with long range", "logwin"),
                    # ("oneref: match one image with all", "oneref")]
                    winsize=1,
                    refid=0,  # Scene Graph: ID
                    win_cyclic=False,  # whether to cyclic
                    TSDF_thresh=0.,
                    shared_intrinsics=True, # Only optimize one set of intrinsics for all views
                )
                print(f"Processing completed. Output saved in {args.output_dir}/{args.seq_name}")
        else:
            main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=False,
                      share=False, gradio_delete_cache=False)



