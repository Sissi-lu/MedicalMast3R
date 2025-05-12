#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from mast3r.model import AsymmetricMASt3R
from mast3r.losses import ConfMatchingLoss, MatchingLoss, APLoss, Regr3D, InfoNCE, Regr3D_ScaleShiftInv
from mast3r.datasets import ARKitScenes, BlendedMVS, Co3d, MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD, ScaredKey, Abs, Servct, SCARED

import mast3r.utils.path_to_dust3r  # noqa
# add mast3r classes to dust3r imports
import dust3r.finetuning
dust3r.finetuning.AsymmetricMASt3R = AsymmetricMASt3R
dust3r.finetuning.Regr3D = Regr3D
dust3r.finetuning.Regr3D_ScaleShiftInv = Regr3D_ScaleShiftInv
dust3r.finetuning.MatchingLoss = MatchingLoss
dust3r.finetuning.ConfMatchingLoss = ConfMatchingLoss
dust3r.finetuning.InfoNCE = InfoNCE
dust3r.finetuning.APLoss = APLoss


import dust3r.datasets
dust3r.datasets.ARKitScenes = ARKitScenes
dust3r.datasets.BlendedMVS = BlendedMVS
dust3r.datasets.Co3d = Co3d
dust3r.datasets.MegaDepth = MegaDepth
dust3r.datasets.ScanNetpp = ScanNetpp
dust3r.datasets.StaticThings3D = StaticThings3D
dust3r.datasets.Waymo = Waymo
dust3r.datasets.WildRGBD = WildRGBD
dust3r.datasets.ScaredKey = ScaredKey
dust3r.datasets.Abs = Abs
dust3r.datasets.Servct = Servct
dust3r.datasets.SCARED = SCARED



from dust3r.finetuning import get_args_parser as dust3r_get_args_parser  # noqa
# from dust3r.training import train  # noqa
from dust3r.finetuning import finetune  # noqa

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_args_parser():
    parser = dust3r_get_args_parser()
    # change defaults
    parser.prog = 'MASt3R finetuning'
    parser.set_defaults(model="AsymmetricMASt3R(patch_embed_cls='ManyAR_PatchEmbed')")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    finetune(args)
