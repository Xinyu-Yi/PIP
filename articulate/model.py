r"""
    SMPL/MANO/SMPLH parametric model. Modified from https://github.com/CalciferZh/SMPL.
"""


__all__ = ['ParametricModel']


import os
import pickle
import torch
import tqdm
import numpy as np
from . import math as M


class ParametricModel:
    r"""
    SMPL/MANO/SMPLH parametric model.
    """
    def __init__(self, official_model_file: str, use_pose_blendshape=False, device=torch.device('cpu')):
        r"""
        Init an SMPL/MANO/SMPLH parametric model.

        :param official_model_file: Path to the official model to be loaded.
        :param use_pose_blendshape: Whether to use the pose blendshape.
        :param device: torch.device, cpu or cuda.
        """
        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self._J_regressor = torch.from_numpy(data['J_regressor'].toarray()).float().to(device)
        self._skinning_weights = torch.from_numpy(data['weights']).float().to(device)
        self._posedirs = torch.from_numpy(data['posedirs']).float().to(device)
        self._shapedirs = torch.from_numpy(np.array(data['shapedirs'])).float().to(device)
        self._v_template = torch.from_numpy(data['v_template']).float().to(device)
        self._J = torch.from_numpy(data['J']).float().to(device)
        self.face = data['f']
        self.parent = data['kintree_table'][0].tolist()
        self.parent[0] = None
        self.use_pose_blendshape = use_pose_blendshape

    def save_obj_mesh(self, vertex_position, file_name='a.obj'):
        r"""
        Export an obj mesh using the input vertex position.

        :param vertex_position: Vertex position in shape [num_vertex, 3].
        :param file_name: Output obj file name.
        """
        with open(file_name, 'w') as fp:
            for v in vertex_position:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.face + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    @staticmethod
    def save_unity_motion(pose: torch.Tensor = None, tran: torch.Tensor = None, output_dir='saved_motions/'):
        r"""
        Write motions into text files used by Unity3D `LoadMotion()`.

        :param pose: Pose tensor in shape [num_frames, *] that can reshape to [num_frame, num_joint, 3, 3]
                    (rotation matrices).
        :param tran: Translation tensor in shape [num_frames, 3] for root positions.
        :param output_dir: Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)

        if pose is not None:
            f = open(os.path.join(output_dir, 'pose.txt'), 'w')
            pose = M.rotation_matrix_to_axis_angle(pose).view(pose.shape[0], -1)
            f.write('\n'.join([','.join(['%.4f' % _ for _ in p]) for p in pose]))
            f.close()

        if tran is not None:
            f = open(os.path.join(output_dir, 'tran.txt'), 'w')
            f.write('\n'.join([','.join(['%.5f' % _ for _ in t]) for t in tran.view(tran.shape[0], 3)]))
            f.close()

    def get_zero_pose_joint_and_vertex(self, shape: torch.Tensor = None):
        r"""
        Get the joint and vertex positions in zero pose. Root joint is aligned at zero.

        :param shape: Tensor for model shapes that can reshape to [batch_size, 10]. Use None for the mean(zero) shape.
        :return: Joint tensor in shape [batch_size, num_joint, 3] and vertex tensor in shape [batch_size, num_vertex, 3]
                 if shape is not None. Otherwise [num_joint, 3] and [num_vertex, 3] assuming the mean(zero) shape.
        """
        if shape is None:
            j, v = self._J - self._J[:1], self._v_template - self._J[:1]
        else:
            shape = shape.view(-1, 10)
            v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template
            j = torch.matmul(self._J_regressor, v)
            j, v = j - j[:, :1], v - j[:, :1]
        return j, v

    def bone_vector_to_joint_position(self, bone_vec: torch.Tensor):
        r"""
        Calculate joint positions in the base frame from bone vectors (position difference of child and parent joint)
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param bone_vec: Bone vector tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Joint position, in shape [batch_size, num_joint, 3].
        """
        return M.bone_vector_to_joint_position(bone_vec, self.parent)

    def joint_position_to_bone_vector(self, joint_pos: torch.Tensor):
        r"""
        Calculate bone vectors (position difference of child and parent joint) in the base frame from joint positions
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param joint_pos: Joint position tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Bone vector, in shape [batch_size, num_joint, 3].
        """
        return M.joint_position_to_bone_vector(joint_pos, self.parent)

    def forward_kinematics_R(self, R_local: torch.Tensor):
        r"""
        :math:`R_global = FK(R_local)`

        Forward kinematics that computes the global rotation of each joint from local rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint global rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return M.forward_kinematics_R(R_local, self.parent)

    def inverse_kinematics_R(self, R_global: torch.Tensor):
        r"""
        :math:`R_local = IK(R_global)`

        Inverse kinematics that computes the local rotation of each joint from global rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_global: Joint global rotation tensor in shape [batch_size, *] that can reshape to
                         [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint local rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return M.inverse_kinematics_R(R_global, self.parent)

    def forward_kinematics_T(self, T_local: torch.Tensor):
        r"""
        :math:`T_global = FK(T_local)`

        Forward kinematics that computes the global homogeneous transformation of each joint from
        local homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_local: Joint local transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint global transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return M.forward_kinematics_T(T_local, self.parent)

    def inverse_kinematics_T(self, T_global: torch.Tensor):
        r"""
        :math:`T_local = IK(T_global)`

        Inverse kinematics that computes the local homogeneous transformation of each joint from
        global homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_global: Joint global transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint local transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return M.inverse_kinematics_T(T_global, self.parent)

    def forward_kinematics(self, pose: torch.Tensor, shape: torch.Tensor = None, tran: torch.Tensor = None,
                           calc_mesh=False):
        r"""
        Forward kinematics that computes the global joint rotation, joint position, and additionally
        mesh vertex position from poses, shapes, and translations. (torch, batch)

        :param pose: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                     [batch_size, num_joint, 3, 3] (rotation matrices).
        :param shape: Tensor for model shapes that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran: Root position tensor in shape [batch_size, 3]. Use None for the zero positions.
        :param calc_mesh: Whether to calculate mesh vertex positions.
        :return: Joint global rotation in [batch_size, num_joint, 3, 3],
                 joint position in [batch_size, num_joint, 3],
                 and additionally mesh vertex position in [batch_size, num_vertex, 3] if calc_mesh is True.
        """
        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        pose = pose.view(pose.shape[0], -1, 3, 3)
        j, v = [_.expand(pose.shape[0], -1, -1) for _ in self.get_zero_pose_joint_and_vertex(shape)]
        T_local = M.transformation_matrix(pose, self.joint_position_to_bone_vector(j))
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = M.decode_transformation_matrix(T_global)
        if calc_mesh is False:
            return pose_global, add_tran(joint_global)

        T_global[..., -1:] -= torch.matmul(T_global, M.append_zero(j, dim=-1).unsqueeze(-1))
        T_vertex = torch.tensordot(T_global, self._skinning_weights, dims=([1], [1])).permute(0, 3, 1, 2)
        if self.use_pose_blendshape:
            r = (pose[:, 1:] - torch.eye(3, device=pose.device)).flatten(1)
            v = v + torch.tensordot(r, self._posedirs, dims=([1], [2]))
        vertex_global = torch.matmul(T_vertex, M.append_one(v, dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, add_tran(joint_global), add_tran(vertex_global)

    def view_joint(self, joint_list: list, fps=60, distance_between_subjects=0.8):
        r"""
        View model joint (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param joint_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joint, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        import vctoolkit as vc
        import vctoolkit.viso3d as vo3d
        joint_list = [(j.view(-1, len(self.parent), 3) - j.view(-1, len(self.parent), 3)[:1, :1]).cpu().numpy()
                      for j in joint_list]

        v_list, f_list = [], []
        f = vc.joints_to_mesh(joint_list[0][0], self.parent)[1]
        for i in range(len(joint_list)):
            v = np.stack([vc.joints_to_mesh(frame, self.parent)[0] for frame in joint_list[i]])
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = np.concatenate(v_list, axis=1)
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.avi', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)

    def view_mesh(self, vertex_list: list, fps=60, distance_between_subjects=0.8):
        r"""
        View model mesh (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param vertex_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_vertex, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        import vctoolkit.viso3d as vo3d
        v_list, f_list = [], []
        f = self.face.copy()
        for i in range(len(vertex_list)):
            v = vertex_list[i].clone().view(-1, self._v_template.shape[0], 3)
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = torch.cat(v_list, dim=1).cpu().numpy()
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.avi', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)

    def view_motion(self, pose_list: list, tran_list: list = None, fps=60, distance_between_subjects=0.8):
        r"""
        View model motion (poses and translations) (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param pose_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joint, 3, 3].
        :param tran_list: List in length [num_subject] of tensors that can all reshape to [num_frame, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        verts = []
        for i in range(len(pose_list)):
            pose = pose_list[i].view(-1, len(self.parent), 3, 3)
            tran = tran_list[i].view(-1, 3) - tran_list[i].view(-1, 3)[:1] if tran_list else None
            verts.append(self.forward_kinematics(pose, tran=tran, calc_mesh=True)[2])
        self.view_mesh(verts, fps, distance_between_subjects=distance_between_subjects)

    def view_mesh_overlay(self, verts, images, K, Tcw=torch.eye(4), fps=60):
        r"""
        View model mesh overlay (single frame or a sequence).
        Warning: open3d has bugs. This function may not run correctly.

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param verts: Tensor that can reshape to [num_frame, num_vertex, 3].
        :param images: Numpy uint8 array that can expand to [num_frame, height, width, 3].
        :param K: Camera intrinsic tensor in shape [3, 3].
        :param Tcw: Camera extrinsic tensor in shape [4, 4].
        :param fps: Sequence FPS.
        """
        import vctoolkit.viso3d as vo3d
        import cv2

        verts = verts.cpu().view(-1, self._v_template.shape[0], 3).numpy()
        images = np.broadcast_to(images, (verts.shape[0], images.shape[-3], images.shape[-2], 3))

        param = vo3d.o3d.camera.PinholeCameraParameters()
        param.intrinsic = vo3d.o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = K.numpy()
        param.extrinsic = Tcw.numpy()

        viewer = vo3d.o3d.visualization.Visualizer()
        viewer.create_window(width=images.shape[-2], height=images.shape[-3], visible=False)
        mesh = vo3d.create_o3d_mesh(verts[0], self.face)
        viewer.add_geometry(mesh)
        view_control = viewer.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        viewer.get_render_option().background_color = [0, 0, 0]
        padw = images.shape[-2] - view_control.convert_to_pinhole_camera_parameters().intrinsic.width
        padh = images.shape[-3] - view_control.convert_to_pinhole_camera_parameters().intrinsic.height
        assert padw >= 0 and padh >= 0
        wb, we = padw // 2, images.shape[-2] + padw // 2 - padw
        hb, he = padh // 2, images.shape[-3] + padh // 2 - padh

        if len(verts) == 1:
            viewer.poll_events()
            viewer.update_renderer()
            frame = (np.asarray(viewer.capture_screen_float_buffer()) * 255).astype(np.uint8)
            mask = np.tile(frame.astype(np.bool8).max(axis=2, keepdims=True), (1, 1, 3))
            im = images[0].copy()
            im[hb:he, wb:we][mask] = frame[mask]
            cv2.imshow('overlay', im)
            cv2.waitKey(0)
        else:
            writer = cv2.VideoWriter('a.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (images[0].shape[1], images[0].shape[0]))
            for i in tqdm.trange(len(verts)):
                mesh.vertices = vo3d.o3d.utility.Vector3dVector(verts[i])
                mesh.compute_vertex_normals()
                viewer.update_geometry(mesh)
                viewer.poll_events()
                viewer.update_renderer()
                frame = (np.asarray(viewer.capture_screen_float_buffer()) * 255).astype(np.uint8)
                mask = np.tile(frame.astype(np.bool8).max(axis=2, keepdims=True), (1, 1, 3))
                im = images[i].copy()
                im[hb:he, wb:we][mask] = frame[mask]
                cv2.imshow('overlay', im)
                cv2.waitKey(1)
                writer.write(im)
            writer.release()
        cv2.destroyWindow('overlay')

    def view_motion_overlay(self, pose, tran, images, K, Tcw=torch.eye(4), fps=60):
        r"""
        View model motion (poses and translations) overlay (single frame or a sequence).
        Warning: open3d has bugs. This function may not run correctly.

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param pose: Tensor that can reshape to [num_frame, num_joint, 3, 3].
        :param tran: Tensor that can reshape to [num_frame, 3].
        :param images: Numpy uint8 array that can expand to [num_frame, height, width, 3].
        :param K: Camera intrinsic tensor in shape [3, 3].
        :param Tcw: Camera extrinsic tensor in shape [4, 4].
        :param fps: Sequence FPS.
        """
        verts = self.forward_kinematics(pose.view(-1, len(self._J), 3, 3), tran=tran.view(-1, 3), calc_mesh=True)[2]
        self.view_mesh_overlay(verts, images, K, Tcw, fps)
