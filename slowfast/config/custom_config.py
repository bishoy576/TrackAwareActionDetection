#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    ## Added by Gurkirt Singh
    ## Additions begin here
    _C.DETECTION.SPATIAL_FPN = False
    _C.DETECTION.ROI_HEAD_TYPE = 'CUBE'
    #### following options added to support multisports, ucf24, and ROAD dataset
    #### and to support different number of classes

    # _C.AVA.MULTILABEL = True
    _C.AVA.SUB_DATASET = 'ava'
    # _C.AVA.POSTFIX = 'ALL'
    # _C.AVA.DETECTION_SCORE_THRESH = 0.01
    _C.AVA.DETECTION_SCORE_THRESH_EVAL = 0.01
    # _C.AVA.TRAIN_PROP_TYPE = "PROP"
    # _C.AVA.TEST_PROP_TYPE = "PROP"
    # _C.AVA.USE_GT_TRAIN = True
    # _C.AVA.USE_GT_TRAIN_ONLY = False
    # _C.AVA.USE_GT_TEST = False
    # _C.AVA.USE_BG_TRAIN = 1
    # _C.AVA.USE_BG_TEST = 1
    # _C.RECONFIGURABLE_CONFIG = False
    _C.WANDB_ENABLE = False
    _C.WANDB_PROJECT = 'Multisports'

    ### Track Related
    _C.AVA.TRACK_ALIGN = False
    _C.AVA.TRACK_TYPES = "gt_yv5SS"
    _C.AVA.TRACKS_THRESH_TRAIN = 0.01
    _C.AVA.TRACKS_THRESH_EVAL  = 0.01
    _C.AVA.INCLUDE_BOXES  = True

    ## Added by Gurkirt Singh End here
    
    return _C
