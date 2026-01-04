import numpy as np

# camera intrinsics
intrinsics = np.eye(3)

# camera extrinsics, mapping gripper frame to camera frame.
X_GC = np.eye(4)
