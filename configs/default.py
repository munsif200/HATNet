from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition for ReST implementation
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# MODEL
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.MODE = 'train'  # 'train' or 'test'
_C.MODEL.RESUME = False
_C.MODEL.LAST_CKPT_FILE = ''

# -----------------------------------------------------------------------------
# SPATIAL_TEMPORAL
_C.ST = CN()
_C.ST.SPATIAL_DIM = 32
_C.ST.TEMPORAL_DIM = 32
_C.ST.HIDDEN_DIM = 64
_C.ST.NUM_LAYERS = 3
_C.ST.DROPOUT = 0.1

# -----------------------------------------------------------------------------
# GRAPH
_C.GRAPH = CN()
_C.GRAPH.NODE_DIM = 32
_C.GRAPH.EDGE_DIM = 6
_C.GRAPH.MESSAGE_DIM = 32
_C.GRAPH.NUM_PASSING_STEPS = 4

# -----------------------------------------------------------------------------
# SOLVER
_C.SOLVER = CN()
_C.SOLVER.TYPE = 'SG'  # 'SG': Spatial Graph, 'TG': Temporal Graph
_C.SOLVER.EPOCHS = 100
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.SCHEDULER = 'cosine'

# -----------------------------------------------------------------------------
# OUTPUT
_C.OUTPUT = CN()
_C.OUTPUT.LOG_DIR = './logs'
_C.OUTPUT.CKPT_DIR = './logs/checkpoints'
_C.OUTPUT.VIS_DIR = './logs/visualizations'

# -----------------------------------------------------------------------------
# TEST
_C.TEST = CN()
_C.TEST.CKPT_FILE_SG = ''
_C.TEST.CKPT_FILE_TG = ''
_C.TEST.EDGE_THRESH = 0.5

cfg = _C