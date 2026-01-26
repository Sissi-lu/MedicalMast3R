# Usage: if you have a model with a pth, and you want to inference within it, you can use this file.
import argparse
import sys
import os
import matplotlib
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dust3r')))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

from mast3r.demo import _convert_scene_output_to_glb, main_demo, SparseGAState, get_3D_model_from_scene
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

# print("CUDA available:", torch.cuda.is_available())--
# print("Device count:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 最近最近

def parse_args():
    parser = argparse.ArgumentParser()
    # where the checkpoints is
    parser.add_argument('--base-dir', type=str, default='/data/luxiaoxi/code_proj/depth_estimation/MedicalMast3R/',
                        help="project location")
    # parser.add_argument('--input-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_depth/eyetube/fundus/test')
    parser.add_argument('--input-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/part3/Dislocated_IOL')

    # parser.add_argument('--input-data', type=str, help='cutting_tissues_twice or pulling_soft_tissues', default='cutting_tissues_twice')


    ## finetune
    # parser.add_argument('--output-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_depth_output/fundus_real/MASt3R_0126_real')
    # parser.add_argument('--model-name', type=str, default='/data_new/luxiaoxi/code_proj/MedicalMast3R/checkpoints/fundusdong_mast3r_1002_embed_head_decoder_true_dataset_test=2000/checkpoint-best.pth')
    # parser.add_argument('--model-name', type=str, default='/data_new/luxiaoxi/code_proj/MedicalMast3R/checkpoints/fundusdong_mast3r_1002_embed_head_decoder_true_dataset_test=2000/checkpoint-best.pth')

    parser.add_argument('--device', type=str, default='cuda')

    ## zeroshot
    parser.add_argument('--output-dir', type=str, default='/data_new/luxiaoxi/dataset/medical_depth_output/fundus_real/MASt3R_0126_zeroshot')
    parser.add_argument('--model-name', type=str, default='/data_new/luxiaoxi/code_proj/MedicalMast3R/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
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



def save_prediction_results(save_folder, scene, clean_depth, min_conf_thr, name):
    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    pts3d[0] = pts3d[0].view(confs[0].shape[0], confs[0].shape[1], -1)
    pts3d[1] = pts3d[1].view(confs[1].shape[0], confs[1].shape[1], -1)



    scene.pts3d = pts3d
    scene.depthmaps = depthmaps
    pts3ds = to_numpy(scene.pts3d)
    np.save(os.path.join(save_folder, "pts3d_left_%s.npy" % name), pts3ds[0])
    np.save(os.path.join(save_folder, "pts3d_right_%s.npy" % name), pts3ds[1])

    poses = scene.save_tum_poses(f'{save_folder}/%s_pred_traj.txt'%name)
    K = scene.save_intrinsics(f'{save_folder}/%s_pred_intrinsics.txt'%name)
    depth_maps = scene.save_depth_maps(f'{save_folder}/%s_depthmaps.txt'%name)
    relative_depth_maps = scene.save_relative_depth_maps(save_folder)

    confidence_masks = to_numpy([c > min_conf_thr for c in confs])

    rgbimg = scene.imgs
    # depths = to_numpy(scene.get_depthmaps())
    depths = to_numpy(scene.depthmaps)
    np.save(os.path.join(save_folder, "preddepth_left_%s.npy" % name), depths[0])
    np.save(os.path.join(save_folder, "preddepth_%s.npy" % name), depths[1])
    depths[0] = depths[0].reshape(confs[0].shape[0], confs[0].shape[1])
    depths[1] = depths[1].reshape(confs[1].shape[0], confs[1].shape[1])


    confs = to_numpy([c for c in confs])

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

    # plt.imshow(depth_imgs[0].reshape(384, 512), cmap='jet')
    # plt.savefig(os.path.join(save_folder, 'depth.png'))

    # np.save(os.path.join(save_folder, "rgb.npy"), rgb_imgs)
    np.save(os.path.join(save_folder, "%s_rgb_depth.npy"%(name)), depth_imgs)
    # np.save(os.path.join(save_folder, "confs.npy"), confs_imgs)

    # ----------------find 2D-2D matches between the two images------------#
    # from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    # pts2d_list, pts3d_list = [], []
    # imgs = scene.imgs
    # for i in range(2):
    #     conf_i = confidence_masks[i]
    #     pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
    #     pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    #
    # try:
    #     reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    #     print(f'found {num_matches} matches')
    #     matches_im1 = pts2d_list[1][reciprocal_in_P2]
    #     matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
    #
    #     # visualize a few matches
    #     if num_matches > 30:
    #         n_viz = 30
    #     else:
    #         n_viz = num_matches
    #     match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    #     viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    #
    #     H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    #     img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    #     img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    #     img_ = np.concatenate((img0, img1), axis=1)
    #     plt.figure()
    #     plt.imshow(img_)
    #     cmap = plt.get_cmap('jet')
    #     for i in range(n_viz):
    #         (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #         plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    #     plt.savefig(os.path.join(save_folder,
    #                              "viz_matches_have_%02d_matches.png" % (num_matches)))
    #     plt.close()
    # except:
    #     txt_dir = os.path.abspath(os.path.join(save_folder, "../.."))
    #     print("%s ==================NO MATCHING==================" % save_folder)
    #     with open(os.path.join(txt_dir, "no_pair_viewer_no_matching.txt"), "a") as f:
    #         f.write("%s \n" % save_folder)
    return rgb_imgs, depth_imgs, confs_imgs

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

    ###################### metric depth ##########################
    # left_depth = left_img.replace("imgs", "metric_depth").replace("png", "npy")
    # right_depth = right_img.replace("imgs", "metric_depth").replace("png", "npy")
    # l_depth = np.load(left_depth)
    # r_depth = np.load(right_depth)

    ###################### relative depth ##########################
    # left_depth = left_img.replace("imgs", "depth")
    # right_depth = right_img.replace("imgs", "depth")
    # l_depth = cv2.imread(left_depth, cv2.IMREAD_GRAYSCALE)
    # r_depth = cv2.imread(right_depth, cv2.IMREAD_GRAYSCALE)



    # right_depth = right_img.replace("img", "depth")
    # assert os.path.exists(left_depth)

    l_img = cv2.imread(left_img)
    r_img = cv2.imread(right_img)

    # l_depth_unchanged = cv2.imread(left_depth, cv2.IMREAD_UNCHANGED)
    # l_depth = cv2.imread(left_depth, cv2.IMREAD_GRAYSCALE)

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
    return left_img, right_img


# def draw_comparison(left_img, left_depth, right_img, right_depth, pred_left, pred_right)

# class SparseGAState:
#     def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
#         self.sparse_ga = sparse_ga
#         self.cache_dir = cache_dir
#         self.outfile_name = outfile_name
#         self.should_delete = should_delete
#
#     def __del__(self):
#         if not self.should_delete:
#             return
#         if self.cache_dir is not None and os.path.isdir(self.cache_dir):
#             shutil.rmtree(self.cache_dir)
#         self.cache_dir = None
#         if self.outfile_name is not None and os.path.isfile(self.outfile_name):
#             os.remove(self.outfile_name)
#         self.outfile_name = None


def predict_depth(save_folder, left_img, right_img, device, model, name):
    # parameters
    current_scene_state = None
    optim_level = "refine"  # choice=["coarse", "refine", "refine+depth"]
    # optim_level = "refine+depth"
    lr1 = 0.07  # Coarese minimum 0.01-0.2
    niter1 = 300  # num_iterations
    lr2 = 0.01  # Fine LR:0.005-0.05
    niter2 = 300  # num_iterations
    min_conf_thr = 1.5  # adjust the confidence threshold0.0-10
    as_pointcloud = True
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    cam_size = 0.2  # adjust the camera size in the output point cloud: 0.001-1
    scenegraph_type = "complete"
    silent = False
    matching_conf_thr = 0 # Matching confidence Thr
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
                                    matching_conf_thr=matching_conf_thr, )

    outfile_name = os.path.join(save_folder, name+"_scene.glb")

    scene_state = SparseGAState(scene, False, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)

    # pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    # scene.depthmaps = depthmaps
    # depths = to_numpy(scene.get_depthmaps())

    rgb_imgs, depth_imgs, confs_imgs = save_prediction_results(save_folder, scene_state.sparse_ga, clean_depth, min_conf_thr, name)


    depths = scene_state.sparse_ga.get_depthmaps()
    # pred_left, pred_right = depths[0].detach().cpu().numpy(), depths[1].detach().cpu().numpy()
    pred_left, pred_right = depth_imgs[0]*255, depth_imgs[1] *255
    return pred_left, pred_right, rgb_imgs, depth_imgs, confs_imgs


def draw_picture(save_folder, pred_left, pred_right, img_left, img_right, name):
    # Load images (assuming these variables are defined)
    left_img = cv2.imread(img_left)
    right_img = cv2.imread(img_right)

    # Create a 3x2 subplot grid
    plt.figure(figsize=(10, 10))  # Adjust figure size as needed

    # Row 1: Original images
    plt.subplot(221)  # 3 rows, 2 columns, position 1
    plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title('Left Image')
    plt.axis('off')  # Optional: hide axes

    plt.subplot(222)  # 3 rows, 2 columns, position 2
    plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title('Right Image')
    plt.axis('off')

    # Row 2: Ground truth depth maps
    plt.subplot(223)  # 3 rows, 2 columns, position 3
    plt.imshow(pred_left, cmap='jet')
    plt.title('Pred Left')
    plt.axis('off')

    # Row 3: Predicted depth maps
    plt.subplot(224)  # 3 rows, 2 columns, position 5
    plt.imshow(pred_right, cmap='jet')
    plt.title('Pred Right')
    plt.axis('off')


    # Adjust layout and save
    plt.tight_layout()  # Prevents overlapping
    plt.show()
    plt.savefig(os.path.join(save_folder, "comparison_figure_%s.png"%(name)), dpi=300)  # Higher DPI for better quality
    plt.close()
    return


# def resize_resolution(pred, target):
#     if not pred.shape == target.shape:
#     # enlarge
#         o_h, o_w = target.shape[:2]
#         pred = pred.reshape(384, 512)
#         pred = cv2.resize(pred, (o_w, o_h), interpolation=cv2.INTER_LANCZOS4)
#         # pred_resized = cv2.GaussianBlur(pred, (5, 5), sigmaX=1.0)
#     # reduce
#         # t_h, t_w = pred.shape
#         # target = cv2.resize(target, (t_w, t_h), interpolation=cv2.INTER_AREA)
#     return pred
def resize_resolution(pred, shape):
    # pred = pred.reshape(shape)
    if not pred.shape == shape:
        o_h, o_w = shape

        pred = cv2.resize(pred, (o_w, o_h))
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

    data_path = os.path.join(args.input_dir, "left")
    img_list = sorted(os.listdir(data_path))

    img_list = [os.path.join(args.input_dir, "left", f) for f in img_list]

    model_name = args.model_name
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
        # save_folder = os.path.join(args.output_dir, img_name)
        save_folder = args.output_dir
        # os.makedirs(save_folder, exist_ok=True)
        # assert os.path.exists(img_path)
        left_img, right_img = load_data_path(img_path)
        img = cv2.imread(left_img)
        img_shape = img.shape[:2]

        try:
            pred_left, pred_right, rgb_imgs, depth_imgs, confs_imgs = predict_depth(save_folder, left_img, right_img, device=args.device, model=model, name=img_name)
        except:
            print('[ERROR] {}'.format(img_name))
            continue
        # pred_left, pred_right, rgb_imgs, depth_imgs, confs_imgs = predict_depth(save_folder, left_img, right_img, device=args.device, model=model, name=img_name)

        pred_left = resize_resolution(pred_left, img_shape)

        plt.imshow(pred_left, cmap='jet')
        plt.savefig(os.path.join(save_folder, 'depth.png'))
        # left_depth = resize_resolution()
        pred_right = resize_resolution(pred_right, img_shape)

        draw_picture(save_folder, pred_left, pred_right, left_img, right_img, img_name)
        def reverse(depth_map):
            result = depth_map.copy()
            d_min = np.min(depth_map)
            d_max = np.max(depth_map)
            if d_max != d_min:
                result = d_max - (depth_map - d_min)
            else:
                return result
            return result

        pred_left = reverse(pred_left)
        lower_bound, upper_bound = np.percentile(pred_left, [5, 95])
        clipped_depth = np.clip(pred_left, lower_bound, upper_bound)

            # Step 3: Robust normalization using median and IQR
            # median = np.median(clipped_depth)
            # iqr = np.percentile(clipped_depth, 75) - np.percentile(clipped_depth, 25)
            # normalized_depth = (clipped_depth - median) / (iqr + 1e-8)  # Avoid division by zero

            # # Optional: Scale to [0, 1] if ground truth is normalized similarly
        pred_left = (clipped_depth - clipped_depth.min()) / (
                clipped_depth.max() - clipped_depth.min() + 1e-8
        )

        # pred_left = (pred_left - pred_left.min()) / (pred_left.max() - pred_left.min())
        jet_cmap = matplotlib.cm.get_cmap('jet')
        pred_image_rgb = jet_cmap(pred_left)[:, :, :3]  # 取 RGB 通道，忽略 alpha
        pred_image_rgb = (pred_image_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型

        # 使用 PIL 保存图像
        pil_image = Image.fromarray(pred_image_rgb)
        pil_image.save(os.path.join(save_folder, 'depth_%s.png' % (img_name)))

        # # ---------------------------evaluation----------------------------#
        # # ---------------------------d1, d2, d3----------------------------#
        # total_metric = dict()
        # # for idx, (pred, gt) in enumerate(zip(pred_left, left_depth)):
        # pred_depth = pred_left
        # gt_depth = left_depth
        # min_depth = 0.001
        # max_depth = 255
        #
        #
        # ##-------------------overlook_shift_and_scared_invariant--------------------#
        #
        # # 假设 eval_depth_numpy 和 total_metric 已定义
        # if min_depth is not None and max_depth is not None:
        #     # 创建 mask：深度值在 min_depth 和 max_depth 之间
        #     mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        #     print(f"  Valid mask pixels: {mask.sum()} / {mask.size}")
        #
        #     # 保存原始形状以生成输出图像
        #     original_shape = pred_depth.shape
        #
        #     # 应用 mask 展平 pred_depth 和 gt_depth
        #     pred_depth = pred_depth[mask]
        #     gt_depth = gt_depth[mask]
        #
        #     # 裁剪 pred_depth 到 [min_depth, max_depth]
        #     pred_depth = pred_depth.copy()  # 避免修改原始数据
        #     pred_depth[pred_depth < min_depth] = min_depth
        #     pred_depth[pred_depth > max_depth] = max_depth
        #
        #     # 计算缩放比例并应用
        #     # ratio = np.median(gt_depth) / (np.median(pred_depth) + 1e-5)
        #     # pred_depth *= ratio
        #     gt_depth, pred_depth = scale_shift_invariant(pred_depth, gt_depth)
        #     pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
        #     gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        #
        #     # 创建与原始形状相同的零数组
        #     pred_image = np.zeros(original_shape, dtype=pred_depth.dtype)
        #
        #     # 将缩放后的 pred_depth 赋值到 mask 对应的位置
        #     pred_image[mask] = pred_depth
        #
        #     # 创建与原始形状相同的零数组用于 gt_image
        #     gt_image = np.zeros(original_shape, dtype=gt_depth.dtype)
        #     # 将 gt_depth 赋值到 mask 对应的位置
        #     gt_image[mask] = gt_depth
        # else:
        #     # 如果没有 min_depth 或 max_depth，直接使用原始 pred_depth 和 gt_depth
        #     pred_image = pred_depth.copy()
        #     gt_image = gt_depth.copy()
        #     pred_depth = pred_depth.copy()
        #     gt_depth = gt_depth.copy()
        #     # 可选择是否应用 ratio 缩放
        #     ratio = np.median(gt_depth) / (np.median(pred_depth) + 1e-5)
        #     pred_depth *= ratio
        #     pred_image = pred_depth
        #
        #
        # # pred 和 gt 用于评估
        # pred = pred_depth
        # gt = gt_depth
        #
        # # 评估深度指标
        # depth_metric = eval_depth_numpy(pred, gt, None)
        #
        # # 更新 total_metric
        # for key, values in depth_metric.items():
        #     total_metric[key] = depth_metric[key]
        #
        # metric_logger.update(**total_metric)
        #
        # # 保存 pred_image 为 jet 颜色映射的图像
        # # 归一化 pred_image 到 [0, 1] 以用于颜色映射
        # if np.max(pred_image) > 0:  # 避免除以零
        #     pred_image_normalized = (pred_image - np.min(pred_image)) / (np.max(pred_image) - np.min(pred_image))
        # else:
        #     pred_image_normalized = pred_image  # 如果全为 0，保持不变
        #
        # # 应用 jet 颜色映射
        # jet_cmap = matplotlib.cm.get_cmap('jet')
        # pred_image_rgb = jet_cmap(pred_image_normalized)[:, :, :3]  # 取 RGB 通道，忽略 alpha
        # pred_image_rgb = (pred_image_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型
        #
        # # 使用 PIL 保存图像
        # pil_image = Image.fromarray(pred_image_rgb)
        # pil_image.save(os.path.join(save_folder,img_name+'_depth.png'))
        #
        # # 保存 gt_image 为 jet 颜色映射的图像
        # # 归一化 gt_image 到 [0, 1] 以用于颜色映射
        # if np.max(gt_image) > 0:  # 避免除以零
        #     gt_image_normalized = (gt_image - np.min(gt_image)) / (np.max(gt_image) - np.min(gt_image))
        # else:
        #     gt_image_normalized = gt_image  # 如果全为 0，保持不变
        #
        # # 应用 jet 颜色映射
        # gt_image_rgb = jet_cmap(gt_image_normalized)[:, :, :3]  # 取 RGB 通道，忽略 alpha
        # gt_image_rgb = (gt_image_rgb * 255).astype(np.uint8)  # 转换为 [0, 255] 的 uint8 类型
        #
        # # 使用 PIL 保存 gt_image
        # pil_image_gt = Image.fromarray(gt_image_rgb)
        # pil_image_gt.save(os.path.join(save_folder, img_name+'_gt.png'))
        #
        # # txt_dir = os.path.abspath(os.path.join(save_folder, ".."))
        # txt_dir = args.output_dir
        # with open(os.path.join(txt_dir, "eval_results.txt"), "a") as f:
        #     f.write("img_path: %s \n" % img_path)
        #     f.write(str(metric_logger))
        #     f.write("\n \n")

if __name__ == "__main__":
    args = parse_args()
    endoscope_evaluation(args)