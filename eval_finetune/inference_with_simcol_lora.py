# Usage: if you have a model with a pth, and you want to inference within it, you can use this file.
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dust3r')))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from eval_scripts.evaluation_matrix import eval_depth, eval_depth_numpy
from dust3r.viz import SceneViz, auto_cam_size
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from collections import defaultdict
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dust3r.demo import get_3D_model_from_scene
from tqdm import tqdm
import json
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
import torch
import shutil

from mast3r.demo import _convert_scene_output_to_glb, main_demo
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
import torch
import mast3r.utils.path_to_dust3r  # noqa
from PIL import Image
from scipy.spatial.transform import Rotation as R
from peft import PeftModel
# torch.cuda.set_device(4)  # Force GPU 4
# device = torch.device("cuda:4")
# print("Forced device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# print("CUDA available:", torch.cuda.is_available())--
# print("Device count:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

def parse_args():
    parser = argparse.ArgumentParser()
    # where the checkpoints is
    parser.add_argument('--base-dir', type=str, default='/data/luxiaoxi/code_proj/depth_estimation/MedicalMast3R/',
                        help="project location")
    parser.add_argument('--model-name', type=str, default='/data_new/luxiaoxi/code_proj/MedicalMast3R/checkpoints/mast3r_simcol_lora_finetune_decoder_encoder_0524/checkpoint-best.pth')
    # where the endoscope is
    parser.add_argument('--input-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_slam/SyntheticColon/')
    # parser.add_argument('--input-data', type=str, help='cutting_tissues_twice or pulling_soft_tissues', default='cutting_tissues_twice')
    parser.add_argument('--output-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_depth_output/SyntheticColon_lora_decoder_encoder_0524/')
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--output-dir', type=str, default='/data/luxiaoxi/dataset/eyetube_phase4_results/anterior/23_Gauge_Plaque_Dissection_of_Anterior_Persistent_Fetal_Vasculature_in_a_2_week_old_Boy/dataset0/dust3r/')
    # parser.add_argument('--model-name', type=str, default='/data/luxiaoxi/code_proj/depth_estimation/MedicalDust3R/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    return parser.parse_args()


def get_gt_poses(scene, root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: all camera poses as quaternion vector and 4x4 projection matrix
    """
    locations = []
    rotations = []
    loc_reader = open(root + '/SavedPosition_' + scene + '.txt', 'r')
    rot_reader = open(root + '/SavedRotationQuaternion_' + scene + '.txt', 'r')
    for line in loc_reader:
        locations.append(list(map(float, line.split())))

    for line in rot_reader:
        rotations.append(list(map(float, line.split())))

    locations = np.array(locations)
    rotations = np.array(rotations)
    poses = np.concatenate([locations, rotations], 1)

    r = R.from_quat(rotations).as_matrix()

    TM = np.eye(4)
    TM[1, 1] = -1

    poses_mat = []
    for i in range(locations.shape[0]):
        ri = r[i]
        Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
        poses_mat.append(Pi_left)

    return np.array(poses_mat)

def get_relative_pose(pose_t0, pose_t1):
    """
    :param pose_tx: 4x4 camera pose describing camera to world frame projection of camera x.
    :return: Position of camera 1's origin in camera 0's frame.
    """
    return np.matmul(np.linalg.inv(pose_t0), pose_t1)

def get_traj(first, P):
    traj, traj_4x4 = [], []
    next = first
    traj.append(next[:3, -1])
    traj_4x4.append(first)

    for i in range(0, P.shape[0]):
        Pi = P[i]
        next = np.matmul(next, Pi)
        traj.append(next[:3, -1])
        traj_4x4.append(next)

    traj = np.array(traj)
    traj_4x4 = np.array(traj_4x4)
    return traj, traj_4x4


def relative_transformation(T_w_cam1, T_w_cam2):
    """
    Compute the relative 4x4 transformation matrix from cam1 to cam2.

    Args:
        T_w_cam1 (np.ndarray): 4x4 transformation matrix from world to cam1.
        T_w_cam2 (np.ndarray): 4x4 transformation matrix from world to cam2.

    Returns:
        np.ndarray: 4x4 transformation matrix from cam1 to cam2.
    """
    # Ensure inputs are 4x4 matrices
    if T_w_cam1.shape != (4, 4) or T_w_cam2.shape != (4, 4):
        raise ValueError("Input matrices must be 4x4")

    # Compute inverse of T_w_cam1
    T_w_cam1_inv = np.linalg.inv(T_w_cam1)

    # Compute relative transformation T_cam1_cam2 = T_w_cam2 * inv(T_w_cam1)
    T_cam1_cam2 = np.dot(T_w_cam2, T_w_cam1_inv)

    return T_cam1_cam2

def online_showing(scene):
    viz = SceneViz()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    min_conf_thr = 0.5
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    valid_mask = to_numpy(scene.get_masks())
    cmap = plt.get_cmap('viridis')
    cam_color = [cmap(i / len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255 * c[0], 255 * c[1], 255 * c[2]) for c in cam_color]

    cam_size = 0.05
    viz.add_pointcloud(pts3d, cam_color, valid_mask)
    for i, pose_c2w, focal in enumerate(cams2world, focals):
        viz.add_camera(pose_c2w=pose_c2w,
                       focal=focal,
                       color=(i * 255, (1 - i) * 255, 0),
                       image=cam_color,
                       cam_size=cam_size)
        viz.show()

    return

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

def save_prediction_results(save_folder, scene, clean_depth, min_conf_thr, img_name):
    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    pts3d[0] = pts3d[0].view(confs[0].shape[0], confs[0].shape[1], -1)
    pts3d[1] = pts3d[1].view(confs[1].shape[0], confs[1].shape[1], -1)

    # scene.pts3d = pts3d
    # scene.depthmaps = depthmaps
    # poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    # K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    # depth_maps = scene.save_depth_maps(save_folder)
    # relative_depth_maps = scene.save_relative_depth_maps(save_folder)
    poses = scene.get_im_poses()

    confidence_masks = to_numpy([c > min_conf_thr for c in confs])

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in confs])
    poses = to_numpy(poses)

    from matplotlib import pyplot as pl
    from dust3r.utils.image import rgb
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    new_confs = [d / confs_max for d in confs]

    rgb_imgs = []
    depth_imgs = []
    confs_imgs = []
    for i in range(len(rgbimg)):
        rgb_imgs.append(rgbimg[i])
        depth_imgs.append(rgb(depths[i]))
        confs_imgs.append(rgb(new_confs[i]))


    # np.save(os.path.join(save_folder, "%s_rgb.npy"%img_name), rgb_imgs)
    np.save(os.path.join(save_folder, "%s_rgb_depth.npy"%img_name), depth_imgs)
    np.save(os.path.join(save_folder, "%s_confs.npy"%img_name), confs_imgs)

    # ----------------find 2D-2D matches between the two images------------#
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    pts2d_list, pts3d_list = [], []
    imgs = scene.imgs
    for i in range(2):
        conf_i = confidence_masks[i]
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])

    try:
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # visualize a few matches
        if num_matches > 30:
            n_viz = 30
        else:
            n_viz = num_matches
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img_ = np.concatenate((img0, img1), axis=1)
        plt.figure()
        plt.imshow(img_)
        cmap = plt.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        plt.savefig(os.path.join(save_folder,
                                 "%s_%02d_matches.png" % (img_name, num_matches)))
        plt.close()
    except:
        txt_dir = os.path.abspath(os.path.join(save_folder, "../.."))
        print("%s ==================NO MATCHING==================" % save_folder)
        with open(os.path.join(txt_dir, "no_pair_viewer_no_matching.txt"), "a") as f:
            f.write("%s \n" % save_folder)
    return rgb_imgs, depth_imgs, confs_imgs, poses

    # plt.show(block=True)

    # ----------------find 2D-2D matches between the two images------------#





# def draw_comparison(left_img, left_depth, right_img, right_depth, pred_left, pred_right)

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


def predict_depth(save_folder, left_img, right_img, device, model, img_name):
    # parameters
    current_scene_state = None
    optim_level = "refine"  # choice=["coarse", "refine", "refine+depth"]
    lr1 = 0.07  # Coarese minimum 0.01-0.2
    niter1 = 300  # num_iterations
    lr2 = 0.008  # Fine LR:0.005-0.05
    niter2 = 300  # num_iterations
    min_conf_thr = 0.15  # adjust the confidence threshold0.0-10
    as_pointcloud = True
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    cam_size = 0.2  # adjust the camera size in the output point cloud: 0.001-1
    scenegraph_type = "complete"
    silent = False
    matching_conf_thr = 5 # Matching confidence Thr
    subsample = 1

    # [("complete: all possible image pairs", "complete"),
    # ("swin: sliding window", "swin"),
    # ("logwin: sliding window with long range", "logwin"),
    # ("oneref: match one image with all", "oneref")]
    winsize = 1
    refid = 0 # Scene Graph: ID
    win_cyclic = False  # whether to cyclic
    TSDF_thresh = 0.
    shared_intrinsics = True

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    batch_size = 1
    schedule = 'cosine'
    lr = 0.0001
    niter = 300
    filelist = [left_img, right_img]
    images = load_images([left_img, right_img], size=512)
    # images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau1.png'], size=512)

    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0



    cache_dir = save_folder
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, subsample=1, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr)

    outfile_name = os.path.join(save_folder, "%s_scene.glb"%img_name)

    scene_state = SparseGAState(scene, False, cache_dir, outfile_name)
    # outfile = get_3D_model_from_scene(save_folder, silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
    #                                   clean_depth, transparent_cams, cam_size, TSDF_thresh)

    rgb_imgs, depth_imgs, confs_imgs, poses = save_prediction_results(save_folder, scene_state.sparse_ga, clean_depth, min_conf_thr, img_name)

    depths = scene_state.sparse_ga.get_depthmaps()
    pred_left, pred_right = depths[0].detach().cpu().numpy(), depths[1].detach().cpu().numpy()

    return pred_left, pred_right, rgb_imgs, depth_imgs, confs_imgs, poses


def draw_picture(save_folder, pred_left, pred_right, img_left, img_right, left_depth, right_depth, img_name):
    # Load images (assuming these variables are defined)
    left_img = cv2.imread(img_left)
    right_img = cv2.imread(img_right)

    # Create a 3x2 subplot grid
    plt.figure(figsize=(10, 10))  # Adjust figure size as needed

    # Row 1: Original images
    plt.subplot(321)  # 3 rows, 2 columns, position 1
    plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title('Left Image')
    plt.axis('off')  # Optional: hide axes

    plt.subplot(322)  # 3 rows, 2 columns, position 2
    plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title('Right Image')
    plt.axis('off')

    # Row 2: Ground truth depth maps
    plt.subplot(323)  # 3 rows, 2 columns, position 3
    plt.imshow(pred_left, cmap='jet')
    plt.title('Pred Left')
    plt.axis('off')

    # Row 3: Predicted depth maps
    plt.subplot(324)  # 3 rows, 2 columns, position 5
    plt.imshow(pred_right, cmap='jet')
    plt.title('Pred Right')
    plt.axis('off')

    # Row 2: Ground truth depth maps
    plt.subplot(325)  # 3 rows, 2 columns, position 3
    plt.imshow(left_depth, cmap='jet')
    plt.title('GT Left')
    plt.axis('off')

    # Row 3: Predicted depth maps
    plt.subplot(326)  # 3 rows, 2 columns, position 5
    plt.imshow(right_depth, cmap='jet')
    plt.title('GT Right')
    plt.axis('off')


    # Adjust layout and save
    plt.tight_layout()  # Prevents overlapping
    plt.show()
    plt.savefig(os.path.join(save_folder, "comparison_figure_%s.png"%img_name), dpi=300)  # Higher DPI for better quality
    plt.close()
    return


def resize_resolution(pred, target):
    if not pred.shape == target.shape:
    # enlarge
        o_h, o_w = target.shape[:2]
        pred = cv2.resize(pred, (o_w, o_h), interpolation=cv2.INTER_LANCZOS4)
        # pred_resized = cv2.GaussianBlur(pred, (5, 5), sigmaX=1.0)
    # reduce
        # t_h, t_w = pred.shape
        # target = cv2.resize(target, (t_w, t_h), interpolation=cv2.INTER_AREA)
    return pred


# 记得用scale shift 来处理代码
def scale_shift_invariant(pred, gt):
    # pred: predicted depth map (H, W), gt: ground truth depth map (H, W)

    # Step 1: Center the depth maps
    mu_pred = np.median(pred)  # Scalar
    mu_gt = np.median(gt)  # Scalar
    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt

    # Step 2: Normalize scale
    # Compute the RMS value of the centered depth maps
    scale_pred = np.sqrt(np.median(pred_centered ** 2))  # Scalar
    scale_gt = np.sqrt(np.median(gt_centered ** 2))  # Scalar
    # Normalize, adding a small epsilon to avoid division by zero
    pred_normalized = pred_centered / (scale_pred + 1e-6)
    gt_normalized = gt_centered / (scale_gt + 1e-6)
    # shift = min(min(pred_normalized), min(gt_normalized))
    # pred_shifted = pred_normalized + np.abs(np.min(pred_normalized)) + 0.0001
    # gt_shifted = gt_normalized + np.abs(np.min(gt_normalized)) + 0.0001

    return gt_normalized, pred_normalized

### GT -> pred 中值对齐

def get_scale(gt, pred):
    scale_factor = np.sum(gt[:, :3, -1] * pred[:, :3, -1])/np.sum(pred[:, :3, -1] ** 2)
    return scale_factor

def compute_translation_errors(gt, pred, delta=1):
    errs = []
    rot_err = []
    rot_gt = []
    trans_gt = []
    for i in range(pred.shape[0]-delta):
        Q = np.linalg.inv(gt[i, :, :]) @ gt[i+delta, :, :]
        P = np.linalg.inv(pred[i, :, :]) @ pred[i+delta, :, :]
        E = np.linalg.inv(Q) @ P
        t = E[:3, -1]
        t_gt = Q[:3, -1]
        trans = np.linalg.norm(t, ord=2)
        errs.append(trans)
        tr = np.arccos((np.trace(E[:3, :3]).clip(-3,3) -1)/2)
        gt_tr = np.arccos((np.trace(Q[:3, :3]) -1)/2)
        rot_err.append(tr)
        rot_gt.append(gt_tr)
        trans_gt.append(np.linalg.norm(t_gt, ord=2))

    errs = np.array(errs)

    ATE = np.median(np.linalg.norm((gt[:, :, -1] - pred[:, :, -1]), ord=2, axis=1))
    RTE = np.median(errs)
    ROT = np.median(rot_err)

    return ATE, RTE, errs, ROT * 180 / np.pi, np.mean(rot_gt) * 180 / np.pi

def endoscope_evaluation(args):
    device = args.device

    base_dir = args.base_dir

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))

    # for folder_idx, folder in enumerate(folder_lists):
    header = 'folder: [{}]'.format("endonerf")

    img_list = []
    img_list = [f for f in os.listdir(args.input_dir) if f.startswith("FrameBuffer")]
    img_list = [os.path.join(args.input_dir, f) for f in img_list]

    model_name = os.path.join(args.base_dir, args.model_name)
    assert os.path.exists(model_name), '{} does not exist'.format(model_name)
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    abs_path = os.path.abspath(os.path.join(args.model_name, ".."))
    lora_path = os.path.join(abs_path, "lora-best")
    # adapter_config_path = os.path.join(args.lora_path, 'adapter_config.json')
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    print(f"Loaded LoRA adapter from {lora_path}")


    idx = 0

    folders = sorted([f for f in os.listdir(args.input_dir) if not f.endswith('txt')])
    for folder in folders:
        print('---------------folder: {}-------------------'.format(folder))
        frames_list = sorted([f for f in os.listdir(os.path.join(args.input_dir, folder)) if not f.endswith('.txt')])
        for frame_folder in frames_list:
            print('===============frame_folder: {}-----------------'.format(frame_folder))
            img_list = sorted([f for f in os.listdir(os.path.join(args.input_dir, folder, frame_folder)) if f.startswith('FrameBuffer')])
            img_list = [os.path.join(args.input_dir, folder, frame_folder, f) for f in img_list]
            save_folder = os.path.join(args.output_dir, folder, frame_folder)
            os.makedirs(save_folder, exist_ok=True)
            ## ---------------Deal with GT poses-----------------##
            sequence = frame_folder.split('_')[1]
            gt_abs_poses = get_gt_poses(sequence, os.path.join(args.input_dir, folder))
            gt_rel_poses = []
            ## Absolute gt poses --> relative gt poses
            delta = 1
            for i in range(0, gt_abs_poses.shape[0] - 1, delta):
                out = get_relative_pose(gt_abs_poses[i], gt_abs_poses[i + delta])
                gt_rel_poses.append(out)
            first_pose = gt_abs_poses[0]

            pred_rel_poses = []
            for img_path in metric_logger.log_every(img_list[:-1], print_freq=1, header=header):
                # if args.data_name.split('_')[0] == "servct":
                #     save_folder = os.path.join(output_dir, img_path.split('.')[0])
                # elif args.data_name.split('_')[0] == "fundus":
                #     save_folder = os.path.join(output_dir, img_path.split('/')[-4], img_path.split('/')[-3],
                #                                img_path.split('/')[-1].split('.')[0])
                # else:
                #     save_folder = os.path.join(output_dir, img_path.split('/')[-2], img_path.split('/')[-1])
                img_name = img_path.split('/')[-1].split('.')[0]

                # left_img: path for i img, right_img: path for i+1 img
                left_img = img_path
                right_img = img_list[idx+1]
                left_depth = np.array(Image.open(left_img.replace('FrameBuffer', 'Depth'))) /255 / 256
                right_depth = np.array(Image.open(right_img.replace('FrameBuffer', 'Depth'))) / 255 /256

                pred_left, pred_right, rgb_imgs, depth_imgs, confs_imgs, poses = predict_depth(save_folder, left_img, right_img, device=args.device, model=model, img_name=img_name)
                pred_rel_pose = relative_transformation(poses[0], poses[1])
                pred_rel_poses.append(pred_rel_pose)


                pred_left = resize_resolution(pred_left, left_depth)
                # left_depth = resize_resolution()
                pred_right = resize_resolution(pred_right, right_depth)

                draw_picture(save_folder, pred_left, pred_right, left_img, right_img, left_depth, right_depth, img_name)

                # ---------------------------evaluation----------------------------#
                # ---------------------------d1, d2, d3----------------------------#
                total_metric = dict()
                # for idx, (pred, gt) in enumerate(zip(pred_left, left_depth)):
                pred_depth = pred_left
                gt_depth = left_depth
                min_depth = 0.001
                max_depth = 20


                ##-------------------overlook_shift_and_scared_invariant--------------------#

                if min_depth is not None and max_depth is not None:
                    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
                    print(f"  Valid mask pixels: {mask.sum()} / {mask.size}")

                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

                pred_depth[pred_depth < min_depth] = min_depth
                pred_depth[pred_depth > max_depth] = max_depth

                ratio = np.median(gt_depth) / (np.median(pred_depth) + 1e-5)
                pred_depth *= ratio

                # pred = (pred - pred.min()) / (pred.max() - pred.min())
                # gt = (gt - gt.min()) / (gt.max() - gt.min())

                depth_metric = eval_depth_numpy(pred_depth, gt_depth, None)


                for key, values in depth_metric.items():
                    total_metric[key] = depth_metric[key]

                metric_logger.update(**total_metric)


                txt_dir = os.path.abspath(os.path.join(save_folder, "../.."))
                with open(os.path.join(txt_dir, "eval_depth_results.txt"), "a") as f:
                    if idx == 0:
                        f.write('-----------Results sequence %s---------\n' % sequence)
                        f.write('length of sequences: %d \n' % len(img_list))
                    f.write(str(metric_logger))
                    f.write("\n")
                    if idx == len(img_list) - 2:
                        f.write('----------------------------------------\n')
                        f.write('\n\n\n')
                idx = idx + 1

            with open(os.path.join(save_folder, "pred_poses.txt"), "w") as f:
                for pose in pred_rel_pose:
                    pose = pose.reshape(-1)
                    line = ' '.join(map(str, pose))
                    f.write(line + '\n')

            gt_traj, gt_traj_4x4 = get_traj(first_pose, np.array(gt_rel_poses))  # This is not necessary, just to show that get_traj() maps relative gt poses back to gt_abs_poses
            pred_traj, pred_traj_4x4 = get_traj(first_pose, np.array(pred_rel_poses))
            scale = get_scale(np.array(gt_rel_poses[:len(pred_rel_poses)]), np.array(pred_rel_poses))
            ATE, RTE, errs, ROT, gt_rot_mag = compute_translation_errors(gt_traj_4x4[:len(pred_traj_4x4)], pred_traj_4x4)
            print('------------------------')
            print('Results sequence ', sequence)
            print('length of sequences:', len(gt_traj_4x4))
            print("Scale {:8.4f}".format(scale))
            print("ATE {:10.4f} cm".format(ATE))
            print("RTE {:10.4f} cm".format(RTE))
            print("ROT {:10.4f} degrees".format(ROT))
            txt_dir = os.path.abspath(os.path.join(save_folder, "../.."))
            with open(os.path.join(txt_dir, "eval_pose_results.txt"), "a") as f:
                f.write('-----------Results sequence %s---------\n'%sequence)
                f.write('length of sequences: %d \n'%len(gt_traj_4x4))
                f.write("Scale {:8.4f} \n".format(scale))
                f.write("ATE {:10.4f} cm \n".format(ATE))
                f.write("RTE {:10.4f} cm \n".format(RTE))
                f.write("ROT {:10.4f} degrees \n".format(ROT))
                f.write("\n\n\n")


if __name__ == "__main__":
    args = parse_args()
    endoscope_evaluation(args)