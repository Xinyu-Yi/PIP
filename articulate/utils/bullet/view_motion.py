r"""
    View human motions in real-time using pybullet.
"""


__all__ = ['MotionViewer']


import time
import pybullet as p
import numpy as np
import os
from .bullet import change_color
import matplotlib
from scipy.spatial.transform import Rotation


_smpl_to_bullet = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32,
                   6, 7, 8, 15, 16, 17, 24, 25, 26, 39, 40, 41, 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67,
                   68, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 33, 34, 35, 42, 43, 44]


class MotionViewer:
    r"""
    View human motions in real-time / offline.
    """
    colors = matplotlib.colormaps['tab10'].colors

    def __init__(self, n=1, overlap=True):
        r"""
        :param n: Number of human motions to simultaneously show.
        """
        self.n = n
        self.physics_client = 0
        self.offsets = [(((n - 1) / 2 - i) * 1.2 if not overlap else 0, 0, 0) for i in range(n)]
        self.subjects = []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), 'models'))
        p.loadURDF('plane.urdf', [0, -0.881, 0.0], [-0.7071068, 0, 0, 0.7071068])
        for i in range(self.n):
            body = p.loadURDF('body.urdf', self.offsets[i], useFixedBase=False, flags=p.URDF_MERGE_FIXED_LINKS)
            change_color(body, self.colors[i])
            self.subjects.append(body)

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
        self.subjects = []

    def update_all(self, poses: list, trans: list):
        r"""
        Update all subject's motions together.

        :param poses: List of pose tensor/ndarray that can all reshape to [24, 3, 3].
        :param trans: List of tran tensor/ndarray that can all reshape to [3].
        """
        assert len(poses) == len(trans) == self.n, 'Number of motions is not equal to the init value in MotionViewer.'
        for i, (pose, tran) in enumerate(zip(poses, trans)):
            self.update(pose, tran, i)

    def update(self, pose, tran, index=0):
        r"""
        Update the ith subject's motion using smpl pose and tran.

        :param pose: Tensor or ndarray that can reshape to [24, 3, 3] for smpl pose.
        :param tran: Tensor or ndarray that can reshape to [3] for smpl tran.
        :param index: The index of the subject to update.
        """
        assert p.isConnected(self.physics_client), 'MotionViewer is not connected.'
        pose = np.array(pose).reshape((24, 3, 3))
        tran = np.array(tran).reshape(3) + np.array(self.offsets[index])
        euler_poses = Rotation.from_matrix(pose[1:]).as_euler('XYZ').reshape(69)[_smpl_to_bullet].reshape(-1, 1)
        euler_glbrots = Rotation.from_matrix(pose[:1]).as_euler('xyz').reshape(3)
        p.resetJointStatesMultiDof(self.subjects[index], list(range(1, p.getNumJoints(self.subjects[index]))), euler_poses)
        p.resetBasePositionAndOrientation(self.subjects[index], tran, p.getQuaternionFromEuler(euler_glbrots))

    def view_offline(self, poses: list, trans: list, fps=60):
        r"""
        View motion sequences offline.

        :param poses: List of pose tensor/ndarray that can all reshape to [N, 24, 3, 3].
        :param trans: List of tran tensor/ndarray that can all reshape to [N, 3].
        :param fps: Sequence fps.
        """
        is_connected = p.isConnected(self.physics_client)
        if not is_connected:
            self.connect()
        for i in range(trans[0].reshape(-1, 3).shape[0]):
            t = time.time()
            self.update_all([r[i] for r in poses], [r[i] for r in trans])
            time.sleep(max(t + 1 / fps - time.time(), 0))
        if not is_connected:
            self.disconnect()
