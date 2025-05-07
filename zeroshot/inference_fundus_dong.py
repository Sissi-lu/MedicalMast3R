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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
# torch.cuda.set_device(4)  # Force GPU 4
# device = torch.device("cuda:4")
# print("Forced device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# print("CUDA available:", torch.cuda.is_available())
# print("Device count:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

def parse_args():
    parser = argparse.ArgumentParser()
    # where the checkpoints is
    parser.add_argument('--base-dir', type=str, default='/data_new/luxiaoxi/code_proj/MedicalMast3R/',
                        help="project location")
    parser.add_argument('--model-name', type=str, default='checkpoints/fundusdong_bs2_new/checkpoint-best.pth')
    # where the endoscope is
    parser.add_argument('--input-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_depth/final_version_processed')
    # parser.add_argument('--input-data', type=str, help='cutting_tissues_twice or pulling_soft_tissues', default='cutting_tissues_twice')
    parser.add_argument('--output-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_depth_output/fundus_dong/Mast3R_new')
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--output-dir', type=str, default='/data/luxiaoxi/dataset/eyetube_phase4_results/anterior/23_Gauge_Plaque_Dissection_of_Anterior_Persistent_Fetal_Vasculature_in_a_2_week_old_Boy/dataset0/dust3r/')
    # parser.add_argument('--model-name', type=str, default='/data/luxiaoxi/code_proj/depth_estimation/MedicalDust3R/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    return parser.parse_args()


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

def save_prediction_results(save_folder, scene, clean_depth, min_conf_thr):
    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    pts3d[0] = pts3d[0].view(confs[0].shape[0], confs[0].shape[1], -1)
    pts3d[1] = pts3d[1].view(confs[1].shape[0], confs[1].shape[1], -1)

    # scene.pts3d = pts3d
    scene.depthmaps = depthmaps

    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    relative_depth_maps = scene.save_relative_depth_maps(save_folder)

    confidence_masks = to_numpy([c > min_conf_thr for c in confs])

    # outfile = get_3D_model_from_scene(save_folder, silent=False, scene=scene, min_conf_thr=1.1,
    #                                   as_pointcloud=True, mask_sky=False,
    #                                   clean_depth=True, transparent_cams=False, cam_size=0.05, show_cam=True,
    #                                   save_name=None)

    # online_showing(scene)

    # rgbs = scene.save_rgb_imgs(save_folder)
    save_poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    save_depth_maps = scene.save_depth_maps(save_folder)
    save_relative_maps = scene.save_relative_depth_maps(save_folder)

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
                                 "viz_matches_have_%02d_matches.png" % (num_matches)))
        plt.close()
    except:
        txt_dir = os.path.abspath(os.path.join(save_folder, "../.."))
        print("%s ==================NO MATCHING==================" % save_folder)
        with open(os.path.join(txt_dir, "no_pair_viewer_no_matching.txt"), "a") as f:
            f.write("%s \n" % save_folder)
    return

    # plt.show(block=True)

    # ----------------find 2D-2D matches between the two images------------#


def load_data_path(img_path):
    ##
    # img_path = img_path.replace("data_new", "data")
    # if data_name.split('_')[0] == "abs":
    #     left_img = os.path.join(img_path, "imgL.png")
    #     right_img = os.path.join(img_path, "imgR.png")
    #
    #     left_depth = left_img.replace("img", "depth")
    #     right_depth = right_img.replace("img", "depth")
    # elif data_name.split('_')[0] == "scared":
    #     left_img = os.path.join(img_path, "Left_Image.png")
    #     right_img = os.path.join(img_path, "Right_Image.png")
    #
    #     left_depth = left_img.replace("Left_Image", "depthmap_left")
    #     right_depth = right_img.replace("Right_Image", "depthmap_right")
    # elif data_name.split('_')[0] == "servct":
    #
    #     data_root = "/data_new/luxiaoxi/dataset/medical_depth/SERV-CT_preprocessed"
    #     # data_root = "/data/luxiaoxi/dataset/medical_depth/SERV-CT_preprocessed"
    #
    #     left_img = os.path.join(data_root, "left", img_path)
    #     right_img = os.path.join(data_root, "right", img_path)
    #
    #     left_depth = left_img.replace("left", "depthL")
    #     right_depth = right_img.replace("right", "depthR")
    # elif data_name.split('_')[0] == "fundus":
    left_img = img_path
    right_img = img_path.replace('left', 'right')

    left_depth = left_img.replace("imgs", "metric_depth").replace("png", "npy")
    right_depth = right_img.replace("imgs", "metric_depth").replace("png", "npy")
    # right_depth = right_img.replace("img", "depth")
    assert os.path.exists(left_depth)

    l_img = cv2.imread(left_img)
    r_img = cv2.imread(right_img)

    # l_depth_unchanged = cv2.imread(left_depth, cv2.IMREAD_UNCHANGED)
    # l_depth = cv2.imread(left_depth, cv2.IMREAD_GRAYSCALE)
    l_depth = np.load(left_depth)
    r_depth = np.load(right_depth)
    # l_depth = cv2.imread(left_depth, cv2.IMREAD_UNCHANGED) / 256.0
    # r_depth = cv2.imread(right_depth, cv2.IMREAD_UNCHANGED) / 256.0
    # plt.subplot(121)
    # plt.imshow(left_)
    # plt.title('left depth')
    # plt.colorbar()
    #
    # plt.subplot(122)
    # plt.imshow(right_)
    # plt.title('right depth')
    # plt.colorbar()
    # plt.savefig(os.path.join(save_folder, "left_and_right_rgbs.png"))
    # plt.close()
    return left_img, l_depth, right_img, r_depth


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


def predict_depth(save_folder, left_img, right_img, device, model):
    # parameters
    current_scene_state = None
    optim_level = "refine+depth"  # choice=["coarse", "refine", "refine+depth"]
    lr1 = 0.07  # Coarese minimum 0.01-0.2
    niter1 = 500  # num_iterations
    lr2 = 0.014  # Fine LR:0.005-0.05
    niter2 = 500  # num_iterations
    min_conf_thr = 0.0001  # adjust the confidence threshold0.0-10
    as_pointcloud = True
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    cam_size = 0.2  # adjust the camera size in the output point cloud: 0.001-1
    scenegraph_type = "complete"
    silent = False
    matching_conf_thr = 5. # Matching confidence Thr

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
    lr = 0.01
    niter = 500
    filelist = [left_img, right_img]
    images = load_images([left_img, right_img], size=512)
    # images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau1.png'], size=512)

    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0


    cache_dir = save_folder
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr,)

    outfile_name = os.path.join(save_folder, "scene.glb")

    scene_state = SparseGAState(scene, False, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(save_folder, silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)

    save_prediction_results(save_folder, scene_state.sparse_ga, clean_depth, min_conf_thr)

    depths = scene_state.sparse_ga.get_depthmaps()
    pred_left, pred_right = depths[0].detach().cpu().numpy(), depths[1].detach().cpu().numpy()

    return pred_left, pred_right


def draw_picture(save_folder, pred_left, pred_right, img_left, img_right, left_depth, right_depth):
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
    plt.savefig(os.path.join(save_folder, "comparison_figure.png"), dpi=300)  # Higher DPI for better quality
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
    mu_pred = np.mean(pred)  # Scalar
    mu_gt = np.mean(gt)  # Scalar
    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt

    # Step 2: Normalize scale
    # Compute the RMS value of the centered depth maps
    scale_pred = np.sqrt(np.mean(pred_centered ** 2))  # Scalar
    scale_gt = np.sqrt(np.mean(gt_centered ** 2))  # Scalar
    # Normalize, adding a small epsilon to avoid division by zero
    pred_normalized = pred_centered / (scale_pred + 1e-6)
    gt_normalized = gt_centered / (scale_gt + 1e-6)
    # pred_shifted = pred_normalized + np.abs(np.min(pred_normalized))
    # gt_shifted = gt_normalized + np.abs(np.min(gt_normalized))

    return gt_normalized, pred_normalized

### GT -> pred 中值对齐


def endoscope_evaluation(args):
    device = args.device

    base_dir = args.base_dir

    # ckpt_dir = os.path.join(args.base_dir, 'checkpoints', args.data_name)

    # with open(os.path.join(ckpt_dir, "list_data.json"), "r") as f:
    #     folders = json.load(f)

    folder_lists = ["folder_%d" % (i) for i in range(5)]

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))

    # for folder_idx, folder in enumerate(folder_lists):
    header = 'folder: [{}]'.format("endonerf")

    img_list = []
    with open(os.path.join(args.input_dir, "split", "test.txt")) as file:
        for line in file.readlines():
            line = line.strip('\n').split(',')[0]
            img_list.append(line)
    img_list = [os.path.join(args.input_dir, f) for f in img_list]

    model_name = os.path.join(args.base_dir, args.model_name)
    assert os.path.exists(model_name), '{} does not exist'.format(model_name)
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    for img_path in metric_logger.log_every(img_list, print_freq=1, header=header):
        # if args.data_name.split('_')[0] == "servct":
        #     save_folder = os.path.join(output_dir, img_path.split('.')[0])
        # elif args.data_name.split('_')[0] == "fundus":
        #     save_folder = os.path.join(output_dir, img_path.split('/')[-4], img_path.split('/')[-3],
        #                                img_path.split('/')[-1].split('.')[0])
        # else:
        #     save_folder = os.path.join(output_dir, img_path.split('/')[-2], img_path.split('/')[-1])
        img_name = img_path.split('/')[-1].split('.')[0]
        save_folder = os.path.join(args.output_dir, img_name)
        os.makedirs(save_folder, exist_ok=True)
        # assert os.path.exists(img_path)
        left_img, left_depth, right_img, right_depth = load_data_path(img_path)

        pred_left, pred_right = predict_depth(save_folder, left_img, right_img, device=args.device, model=model)

        pred_left = resize_resolution(pred_left, left_depth)
        # left_depth = resize_resolution()
        # pred_right = resize_resolution(pred_right, right_depth)

        draw_picture(save_folder, pred_left, pred_right, left_img, right_img, left_depth, right_depth)

        # ---------------------------evaluation----------------------------#
        # ---------------------------d1, d2, d3----------------------------#
        total_metric = dict()
        # for idx, (pred, gt) in enumerate(zip(pred_left, left_depth)):
        pred = pred_left
        gt = left_depth


        ##-------------------overlook_shift_and_scared_invariant--------------------#
        gt, pred = scale_shift_invariant(pred, gt)

        pred = (pred - pred.min()) / (pred.max() - pred.min())
        # # # pred = 1/(1e-6 + pred)
        # # # gt = 1/(1e-6 + gt)
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        # draw_picture(save_folder, pred, gt, left_img, right_img)
        depth_metric = eval_depth_numpy(pred, gt, None)


        for key, values in depth_metric.items():
            total_metric[key] = depth_metric[key]

        metric_logger.update(**total_metric)


        txt_dir = os.path.abspath(os.path.join(save_folder, ".."))
        with open(os.path.join(txt_dir, "eval_results.txt"), "a") as f:
            f.write("img_path: %s \n" % img_path)
            f.write(str(metric_logger))
            f.write("\n \n")

if __name__ == "__main__":
    args = parse_args()
    endoscope_evaluation(args)