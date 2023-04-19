r"""
    View 3D rotations in real-time using pybullet.
"""


__all__ = ['RotationViewer']


import time
import pybullet as p
import pybullet_data
import torch
import numpy as np
import cv2
from typing import List, Tuple, Union


class RotationViewer:
    r"""
    View 3D rotations in real-time / offline.
    """

    camera_distance = 5

    def __init__(self, n=1, overlap=False):
        r"""
        :param n: Number of rotations to simultaneously show.
        """
        self.n = n
        self.interval = 0 if overlap else 1.2
        self.physics_client = 0
        self.objs = []

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
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=0, cameraPitch=-30,
                                     cameraTargetPosition=[0, 0, 0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        offset = (self.n - 1) * self.interval / 2
        self.objs = [p.loadURDF('duck_vhacd.urdf', [-i * self.interval + offset, 0, 0], globalScaling=10) for i in range(self.n)]

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)

    def update_all(self, rotations: Union[List[torch.Tensor], Tuple[torch.Tensor]]):
        r"""
        Update all rotations together.

        :param rotations: List of Tensors in shape [3, 3] for rotation matrices.
        """
        assert len(rotations) == self.n, 'Number of rotations is not equal to the init value in RotationViewer.'
        for i, rotation in enumerate(rotations):
            self.update(rotation, i)

    def update(self, rotation: torch.Tensor, index=0):
        r"""
        Update the ith rotation.

        :param rotation: Tensor in shape [3, 3] for a rotation matrix.
        :param index: The index of the rotation to update.
        """
        assert p.isConnected(self.physics_client), 'RotationViewer is not connected.'
        position, _ = p.getBasePositionAndOrientation(self.objs[index])
        rotation = cv2.Rodrigues(rotation.clone().detach().cpu().view(3, 3).numpy())[0].ravel()
        angle = np.linalg.norm(rotation)
        axis = rotation / angle if angle != 0 else np.array([1, 0, 0])
        p.resetBasePositionAndOrientation(self.objs[index], position, p.getQuaternionFromAxisAngle(axis, angle))

    def view_offline(self, rotations: Union[List[torch.Tensor], Tuple[torch.Tensor]], fps=60):
        r"""
        View 3D rotation sequences offline.

        :param rotations: List of Tensors in shape [seq_len, 3, 3] for rotation matrices.
        :param fps: Sequence fps.
        """
        rotations = [r.view(-1, 3, 3) for r in rotations]
        seq_len = rotations[0].shape[0]
        assert all([r.shape[0] == seq_len for r in rotations]), 'Different sequence lengths for RotationViewer'
        is_connected = p.isConnected(self.physics_client)
        if not is_connected:
            self.connect()
        for i in range(seq_len):
            t = time.time()
            self.update_all([r[i] for r in rotations])
            time.sleep(max(t + 1 / fps - time.time(), 0))
        if not is_connected:
            self.disconnect()


# example
if __name__ == '__main__':
    rotations1 = torch.tensor([p.getMatrixFromQuaternion(q) for q in np.linspace([0, 0, 0, 1], [1, 0, 0, 0], num=60)])
    rotations2 = torch.tensor([p.getMatrixFromQuaternion(q) for q in np.linspace([0, 0, 0, 1], [0, 1, 0, 0], num=60)])

    # offline
    RotationViewer(2).view_offline([rotations1, rotations2], fps=30)

    # online
    with RotationViewer(2) as viewer:
        for r1, r2 in zip(rotations1, rotations2):
            viewer.update_all([r1, r2])
            time.sleep(1 / 30)
