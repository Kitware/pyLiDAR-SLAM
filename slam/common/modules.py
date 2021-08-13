import logging

# ----------------------------------------------------------------------------------------------------------------------
# OpenCV
try:
    import cv2

    _with_cv2 = True
except ModuleNotFoundError:
    logging.warning("OpenCV (cv2 python module) not found, visualization disabled")
    _with_cv2 = False

# ----------------------------------------------------------------------------------------------------------------------
# viz3d
try:
    import viz3d

    _with_viz3d = True
except ImportError:
    _with_viz3d = False

# ----------------------------------------------------------------------------------------------------------------------
# open3d
try:
    import open3d

    _with_o3d = True
except ImportError:
    logging.warning("Open3D (open3d python module) not found, some features will be disabled")
    _with_o3d = False

# ----------------------------------------------------------------------------------------------------------------------
# g2o
try:
    import g2o

    _with_g2o = True
except ImportError:
    logging.warning("G2O (g2o python module) not found, some features will be disabled")
    _with_g2o = False
