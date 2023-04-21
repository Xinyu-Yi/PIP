import torch
import numpy as np
import pybullet as p
import articulate as art
from articulate.utils.bullet import *
from articulate.utils.rbdl import *
from utils import *
from qpsolvers import solve_qp
from config import paths


class PhysicsOptimizer:
    test_contact_joints = ['LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2',
                           'SPINE3', 'LSHOULDER', 'RSHOULDER', 'HEAD',
                           'LELBOW', 'RELBOW', 'LHAND', 'RHAND', 'LFOOT', 'RFOOT'
                           ]  # 'LANKLE', 'RANKLE', 'NECK', 'LWRIST', 'RWRIST', 'LCLAVICLE', 'RCLAVICLE'

    def __init__(self, debug=False):
        mu = 0.6
        supp_poly_size = 0.2
        self.debug = debug
        self.model = RBDLModel(paths.physics_model_file, update_kinematics_by_hand=True)
        self.params = read_debug_param_values_from_json(paths.physics_parameter_file)
        self.friction_constraint_matrix = np.array([[np.sqrt(2), -mu, 0],
                                                    [-np.sqrt(2), -mu, 0],
                                                    [0, -mu, np.sqrt(2)],
                                                    [0, -mu, -np.sqrt(2)]])
        self.support_polygon = np.array([[-supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [-supp_poly_size / 2,  0,   supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,   supp_poly_size / 2]])

        if debug:
            p.connect(p.GUI)
            p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
            self.id_robot = p.loadURDF(paths.physics_model_file, [0, 0, 0], useFixedBase=False, flags=p.URDF_MERGE_FIXED_LINKS)
            change_color(self.id_robot, [198 / 255, 238 / 255, 0, 1.0])
            p.loadURDF(paths.plane_file, [0, -0.881, 0.0], [-0.7071068, 0, 0, 0.7071068])
            load_debug_params_into_bullet_from_json(paths.physics_parameter_file)

        # states
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.reset_states()

    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)

    def optimize_frame(self, pose, jvel, contact, acc, return_grf=False):
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]
        v_ref = jvel.numpy()
        c_ref = contact.sigmoid().numpy()
        a_ref = acc.numpy()
        q = self.q
        qdot = self.qdot

        if q is None:
            self.q = q_ref
            if return_grf:
                return pose, torch.zeros(3), [], None
            else:
                return pose, torch.zeros(3)

        # determine the contact joints and points
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        Js = [np.empty((0, self.model.qdot_size))]
        collision_points, collision_joints = [], []
        for joint_name in self.test_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            if joint_id == Body.LFOOT and c_ref[0] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03 or \
               joint_id == Body.RFOOT and c_ref[1] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03 or \
               pos[1] <= self.params['floor_y']:
                collision_joints.append(joint_name)
                for ps in self.support_polygon + pos:
                    collision_points.append(ps)
                    pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
                    Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
        Js = np.vstack(Js)
        nc = len(collision_points)

        # minimize   ||A1 * qddot - b1||^2     for A1, b1 in zip(As1, bs1)
        #            + ||A2 * lambda - b2||^2  for A2, b2 in zip(As2, bs2)
        #            + ||A3 * tau - b3||^2     for A3, b3 in zip(As3, bs3)
        # s.t.       G1 * qddot <= h1          for G1, h1 in zip(Gs1, hs1)
        #            G2 * lambda <= h2         for G2, h2 in zip(Gs2, hs2)
        #            G3 * tau <= h3            for G3, h3 in zip(Gs3, hs3)
        #            A_ * x = b_
        As1, bs1, As2, bs2, As3, bs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        A_, b_ = None, None

        # joint angle PD controller
        if True:
            A = np.hstack((np.zeros((self.model.qdot_size - 3, 3)), np.eye((self.model.qdot_size - 3))))
            b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
            As1.append(A)  # 72 * 75
            bs1.append(b)  # 72

        # joint position PD controller (using root velocity + ref pose to determine target joint position)
        if False:
            for joint_name in ['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                               'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                               'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND']:
                joint_id = vars(Body)[joint_name]
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
                cur_pos = self.model.calc_body_position(q, joint_id)
                tar_pos = self.model.calc_body_position(q_ref, joint_id) - q_ref[:3] + q[:3] + v_ref[0] * self.params['delta_t']
                a_des = 3600 * (tar_pos - cur_pos) - 60 * cur_vel
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
                As1.append(A * 2)
                bs1.append(b * 2)

        # joint position PD controller (using joint velocity to determine target joint position)
        if True:
            for joint_name, v in zip(['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                                      'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                                      'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST'], v_ref[:22]):
                joint_id = vars(Body)[joint_name]
                if joint_id == Body.LFOOT or joint_id == Body.RFOOT: continue
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
                a_des = self.params['kp_linear'] * v * self.params['delta_t'] - self.params['kd_linear'] * cur_vel
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
                As1.append(A * self.params['coeff_jvel'])
                bs1.append(b * self.params['coeff_jvel'])

        # joint velocity (without Jdot * qdot term)
        if False:
            for joint_name, v in zip(
                    ['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                     'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                     'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND'], v_ref):
                joint_id = vars(Body)[joint_name]
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = (-self.model.calc_point_velocity(q, qdot, joint_id) + v) / self.params['delta_t']
                As1.append(A * 2)
                bs1.append(b * 2)

        # IMU acceleration
        if False:
            for joint_name, a in zip(['LWRIST', 'RWRIST', 'LKNEE', 'RKNEE', 'HEAD', 'ROOT'], a_ref):
                joint_id = vars(Body)[joint_name]
                offset = np.zeros(3)
                A = self.model.calc_point_Jacobian(q, joint_id, offset)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(self.model.qdot_size), joint_id, offset) + a
                bs1.append(b * self.params['coeff_acc'])
                As1.append(A * self.params['coeff_acc'])

        # lambda size
        if False:
            As2.append(np.eye(nc * 3) * self.params['coeff_lambda_old'])
            bs2.append(np.zeros(nc * 3))

        # Signorini’s conditions of lambda
        if True:
            if nc != 0:
                A = [np.eye(3) * max(cp[1] - self.params['floor_y'], 0.005) for cp in collision_points]
                A = art.math.block_diagonal_matrix_np(A)
                As2.append(A * self.params['coeff_lambda'])
                bs2.append(np.zeros(nc * 3))

        # tau size
        if True:
            As3.append(art.math.block_diagonal_matrix_np([
                np.eye(6) * self.params['coeff_virtual'],
                np.eye(self.model.qdot_size - 6) * self.params['coeff_tau']
            ]))
            bs3.append(np.zeros(self.model.qdot_size))

        # contacting body joint velocity
        if True:
            for joint_name in self.test_contact_joints[:-2]:
                joint_id = vars(Body)[joint_name]
                pos = self.model.calc_body_position(q, joint_id)
                if pos[1] <= self.params['floor_y']:
                    J = self.model.calc_point_Jacobian(q, joint_id)
                    v = self.model.calc_point_velocity(q, qdot, joint_id)
                    Gs1.append(-self.params['delta_t'] * J)
                    hs1.append(v - [-1e-1, 0, -1e-1])
                    Gs1.append(self.params['delta_t'] * J)
                    hs1.append(-v + [1e-1, 1e2, 1e-1])

        # contacting foot velocity
        if True:
            for joint_name, stable in zip(['LFOOT', 'RFOOT'], c_ref):
                joint_id = vars(Body)[joint_name]
                pos = self.model.calc_body_position(q, joint_id)
                J = self.model.calc_point_Jacobian(q, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id)

                th = -np.log(min(stable, 0.84999) / 0.85)
                th_y = (self.params['floor_y'] - pos[1]) / self.params['delta_t']
                Gs1.append(-self.params['delta_t'] * J)
                hs1.append(v - [-th, th_y, -th])
                Gs1.append(self.params['delta_t'] * J)
                hs1.append(-v + [th, max(th, th_y) + 1e-6, th])

        # GRF friction cone constraint
        if True:
            if nc > 0:
                Gs2.append(art.math.block_diagonal_matrix_np([self.friction_constraint_matrix] * nc))
                hs2.append(np.zeros(nc * 4))

        # equation of motion (equality constraint)
        if True:
            M = self.model.calc_M(q)
            h = self.model.calc_h(q, qdot)
            A_ = np.hstack((-M, Js.T, np.eye(self.model.qdot_size)))
            b_ = h

        As1, bs1, As2, bs2, As3, bs3 = np.vstack(As1), np.concatenate(bs1), np.vstack(As2), np.concatenate(bs2), np.vstack(As3), np.concatenate(bs3)
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = np.vstack(Gs1), np.concatenate(hs1), np.vstack(Gs2), np.concatenate(hs2), np.vstack(Gs3), np.concatenate(hs3)
        G_ = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        h_ = np.concatenate((hs1, hs2, hs3))
        P_ = art.math.block_diagonal_matrix_np([np.dot(As1.T, As1), np.dot(As2.T, As2), np.dot(As3.T, As3)])
        q_ = np.concatenate((-np.dot(As1.T, bs1), -np.dot(As2.T, bs2), -np.dot(As3.T, bs3)))

        # fast solvers are less accurate/robust, and may fail
        init = self.last_x if False and len(self.last_x) == len(q_) else None
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver='quadprog', initvals=init)

        # if x is None or np.linalg.norm(x) > 10000:
        #     x = solve_qp(P_, q_, G_, h_, A_, b_, solver='cvxopt', initvals=init)

        if x is None or np.linalg.norm(x) > 10000:
            print('Warning: QP infeasible. Ignoring Gx <= h constraints')
            x = solve_qp(P_, q_, None, None, A_, b_, solver='quadprog', initvals=init)

        qddot = x[:self.model.qdot_size]
        GRF = x[self.model.qdot_size:-self.model.qdot_size]
        tau = x[-self.model.qdot_size:]

        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        self.q = q
        self.qdot = qdot
        self.last_x = x

        if self.debug:
            # self.clock.tick(60)   # please install pygame
            set_pose(self.id_robot, q)
            self.params = read_debug_param_values_from_bullet()

            if False:   # visualize GRF (no smoothing)
                p.removeAllUserDebugItems()
                for point, force in zip(collision_points, GRF.reshape(-1, 3)):
                    p.addUserDebugLine(point, point + force * 1e-2, [1, 0, 0])

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]
        if not return_grf:
            return pose_opt, tran_opt
        else:
            cj = [vars(art.SMPLJoint)[_].value for _ in collision_joints]
            grf = torch.from_numpy(GRF).float().view(-1, 4, 3).sum(dim=1) if len(cj) > 0 else None
            return pose_opt, tran_opt, cj, grf

