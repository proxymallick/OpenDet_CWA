from detectron2.config import CfgNode as CN


def add_opendet_config(cfg):
    _C = cfg

    # unknown probability loss
    _C.UPLOSS = CN()
    _C.UPLOSS.START_ITER = 100  # usually the same as warmup iter
    _C.UPLOSS.SAMPLING_METRIC = "min_score"
    _C.UPLOSS.TOPK = 3
    _C.UPLOSS.ALPHA = 1.0
    _C.UPLOSS.WEIGHT = 0.5

    # instance contrastive loss
    _C.ICLOSS = CN()
    _C.ICLOSS.OUT_DIM = 128
    _C.ICLOSS.QUEUE_SIZE = 256
    _C.ICLOSS.IN_QUEUE_SIZE = 16
    _C.ICLOSS.BATCH_IOU_THRESH = 0.5
    _C.ICLOSS.QUEUE_IOU_THRESH = 0.7
    _C.ICLOSS.TEMPERATURE = 0.1
    _C.ICLOSS.WEIGHT = 0.21

    # register RoI output layer
    _C.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS = "FastRCNNOutputLayers"
    # known classes
    _C.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = 20
    _C.MODEL.RETINANET.NUM_KNOWN_CLASSES = 20
    # thresh for visualization results.
    _C.MODEL.ROI_HEADS.VIS_IOU_THRESH = 1.0
    # scale for cosine classifier
    _C.MODEL.ROI_HEADS.COSINE_SCALE = 20

    # swin transformer
    _C.MODEL.SWINT = CN()
    _C.MODEL.SWINT.EMBED_DIM = 96
    _C.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    _C.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    _C.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    _C.MODEL.SWINT.WINDOW_SIZE = 7
    _C.MODEL.SWINT.MLP_RATIO = 4
    _C.MODEL.SWINT.DROP_PATH_RATE = 0.2
    _C.MODEL.SWINT.APE = False
    _C.MODEL.BACKBONE.FREEZE_AT = -1
    _C.MODEL.FPN.TOP_LEVELS = 2

    # solver, e.g., adamw for swin
    _C.SOLVER.OPTIMIZER = 'SGD'
    _C.SOLVER.BETAS = (0.9, 0.999)


    # ---------------------------------------------------------------------------- #
    # Additional Configs
    # ---------------------------------------------------------------------------- #
    _C.MODEL.MOBILENET = False
    _C.MODEL.BACKBONE.ANTI_ALIAS = False
    _C.MODEL.RESNETS.DEFORM_INTERVAL = 1
    _C.INPUT.HFLIP_TRAIN = True
    _C.INPUT.CROP.CROP_INSTANCE = True
    _C.INPUT.IS_ROTATE = False

    # ---------------------------------------------------------------------------- #
    # FCOS Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.FCOS = CN()

    # This is the number of foreground classes.
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.FCOS.PRIOR_PROB = 0.01
    _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    _C.MODEL.FCOS.NMS_TH = 0.6
    _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 2000 #1000
    _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000 #1000
    _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 1000 #100
    _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 1000 #100
    _C.MODEL.FCOS.TOP_LEVELS = 2
    _C.MODEL.FCOS.NORM = "GN"  # Support GN or none
    _C.MODEL.FCOS.USE_SCALE = True

    # The options for the quality of box prediction
    # It can be "ctrness" (as described in FCOS paper) or "iou"
    # Using "iou" here generally has ~0.4 better AP on COCO
    # Note that for compatibility, we still use the term "ctrness" in the code
    _C.MODEL.FCOS.BOX_QUALITY = "ctrness"

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    _C.MODEL.FCOS.THRESH_WITH_CTR = False

    # Focal loss parameters
    _C.MODEL.FCOS.LOSS_ALPHA = 0.25
    _C.MODEL.FCOS.LOSS_GAMMA = 2.0

    # The normalizer of the classification loss
    # The normalizer can be "fg" (normalized by the number of the foreground samples),
    # "moving_fg" (normalized by the MOVING number of the foreground samples),
    # or "all" (normalized by the number of all samples)
    _C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
    _C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

    _C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    _C.MODEL.FCOS.USE_RELU = True
    _C.MODEL.FCOS.USE_DEFORMABLE = False

    # the number of convolutions used in the cls and bbox tower
    _C.MODEL.FCOS.NUM_CLS_CONVS = 4
    _C.MODEL.FCOS.NUM_BOX_CONVS = 4
    _C.MODEL.FCOS.NUM_SHARE_CONVS = 0
    _C.MODEL.FCOS.CENTER_SAMPLE = True
    _C.MODEL.FCOS.POS_RADIUS = 1.5
    _C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    _C.MODEL.FCOS.YIELD_PROPOSAL = True
    _C.MODEL.FCOS.YIELD_BOX_FEATURES = True