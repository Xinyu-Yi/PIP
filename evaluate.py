import torch
import tqdm
from config import *
from utils import *
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import articulate as art
from articulate.utils.rbdl import *
from net import PIP


torch.set_printoptions(sci_mode=False)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.figure(dpi=200)
plt.grid(linestyle='-.')
plt.xlabel('Real travelled distance (m)', fontsize=16)
plt.ylabel('Mean translation error (m)', fontsize=16)
plt.title('Cumulative Translation Error', fontsize=18)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReducedPoseEvaluator:
    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]), device=device)
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])

    def __call__(self, pose_p, pose_t, tran_p, tran_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 1000])


class FullPoseEvaluator:
    names = ['Absolute Jitter Error (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator(paths.smpl_file, device=device)

    def __call__(self, pose_p, pose_t, tran_p, tran_t):
        # errs = self._base_motion_loss_fn(pose_p=pose_p[:-1], pose_t=pose_t[:-1], tran_p=tran_p[:-1], tran_t=tran_t[:-1])  # bad data -1
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[4] / 1000])


def evaluate_zmp_distance(poses, trans, fps=60, foot_radius=0.1):
    qs = smpl_to_rbdl(poses, trans)
    qdots = np.empty_like(qs)
    qdots[1:, :3] = (qs[1:, :3] - qs[:-1, :3]) * fps
    qdots[1:, 3:] = art.math.angle_difference(qs[1:, 3:], qs[:-1, 3:]) * fps
    qdots[0] = qdots[1]
    qddots = (qdots[1:] - qdots[:-1]) * fps
    qddots = np.concatenate((qddots[:1], qddots))
    rbdl_model = RBDLModel(paths.physics_model_file)

    floor_height = []
    for q in qs[2:30]:
        lp = rbdl_model.calc_body_position(q, Body.LFOOT)
        rp = rbdl_model.calc_body_position(q, Body.RFOOT)
        floor_height.append(lp[1])
        floor_height.append(rp[1])
    floor_height = torch.tensor(floor_height).mean() + 0.01

    dists = []
    for q, qdot, qddot in zip(qs, qdots, qddots):
        lp = rbdl_model.calc_body_position(q, Body.LFOOT)
        rp = rbdl_model.calc_body_position(q, Body.RFOOT)
        if lp[1] > floor_height and rp[1] > floor_height:
            continue

        zmp = rbdl_model.calc_zero_moment_point(q, qdot, qddot)
        ap = (zmp - lp)[[0, 2]]
        ab = (rp - lp)[[0, 2]]
        bp = (zmp - rp)[[0, 2]]
        if lp[1] <= floor_height and rp[1] <= floor_height:
            # point to line segment distance
            r = (ap * ab).sum() / (ab * ab).sum()
            if r < 0:
                d = np.linalg.norm(ap)
            elif r > 1:
                d = np.linalg.norm(bp)
            else:
                d = np.sqrt((ap * ap).sum() - r * r * (ab * ab).sum())
        else:
            # point to point distance
            d = np.linalg.norm(ap if lp[1] <= floor_height else bp)
        dists.append(max(d - foot_radius, 0))

    return sum(dists) / len(dists)


def run_pipeline(net, data_dir, sequence_ids=None):
    r"""
    Run `net` using the imu data loaded from `data_dir`.
    Save the estimated [Pose[num_frames, 24, 3, 3], Tran[num_frames, 3]] for each of `sequence_ids`.
    """
    print('Loading imu data from "%s"' % data_dir)
    accs, rots, poses, _ = torch.load(os.path.join(data_dir, 'test.pt')).values()
    init_poses = [art.math.axis_angle_to_rotation_matrix(_[0]) for _ in poses]

    data_name = os.path.basename(data_dir)
    output_dir = os.path.join(paths.result_dir, data_name, net.name)
    os.makedirs(output_dir, exist_ok=True)

    if sequence_ids is None:
        sequence_ids = list(range(len(accs)))

    print('Saving the results at "%s"' % output_dir)
    for i in tqdm.tqdm(sequence_ids):
        torch.save(net.predict(accs[i], rots[i], init_poses[i]), os.path.join(output_dir, '%d.pt' % i))


def evaluate(net, data_dir, sequence_ids=None, flush_cache=False, pose_evaluator=ReducedPoseEvaluator(),
             evaluate_pose=False, evaluate_tran=False, evaluate_zmp=False):
    r"""
    Evaluate poses and translations of `net` on all sequences in `sequence_ids` from `data_dir`.
    `net` should implement `net.name` and `net.predict(glb_acc, glb_rot)`.
    """
    data_name = os.path.basename(data_dir)
    result_dir = os.path.join(paths.result_dir, data_name, net.name)
    print_title('Evaluating "%s" on "%s"' % (net.name, data_name))

    _, _, pose_t_all, tran_t_all = torch.load(os.path.join(data_dir, 'test.pt')).values()

    if sequence_ids is None:
        sequence_ids = list(range(len(pose_t_all)))
    if flush_cache and os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    missing_ids = [i for i in sequence_ids if not os.path.exists(os.path.join(result_dir, '%d.pt' % i))]
    cached_ids = [i for i in sequence_ids if os.path.exists(os.path.join(result_dir, '%d.pt' % i))]
    print('Cached ids: %s\nMissing ids: %s' % (cached_ids, missing_ids))
    if len(missing_ids) > 0:
        run_pipeline(net, data_dir, missing_ids)

    pose_errors = []
    tran_errors = {window_size: [] for window_size in list(range(1, 8))}
    zmp_errors = []
    for i in tqdm.tqdm(sequence_ids):
        result = torch.load(os.path.join(result_dir, '%d.pt' % i))
        pose_p, tran_p = result[0], result[1]
        pose_t, tran_t = pose_t_all[i], tran_t_all[i]
        if evaluate_pose:
            pose_t = art.math.axis_angle_to_rotation_matrix(pose_t).view_as(pose_p)
            pose_errors.append(pose_evaluator(pose_p, pose_t, tran_p, tran_t))
        if evaluate_tran:
            # compute gt move distance at every frame
            move_distance_t = torch.zeros(tran_t.shape[0])
            v = (tran_t[1:] - tran_t[:-1]).norm(dim=1)
            for j in range(len(v)):
                move_distance_t[j + 1] = move_distance_t[j] + v[j]

            for window_size in tran_errors.keys():
                # find all pairs of start/end frames where gt moves `window_size` meters
                frame_pairs = []
                start, end = 0, 1
                while end < len(move_distance_t):
                    if move_distance_t[end] - move_distance_t[start] < window_size:
                        end += 1
                    else:
                        if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                            frame_pairs.append((start, end))
                        start += 1

                # calculate mean distance error
                errs = []
                for start, end in frame_pairs:
                    vel_p = tran_p[end] - tran_p[start] 
                    vel_t = tran_t[end] - tran_t[start]
                    errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                if len(errs) > 0:
                    tran_errors[window_size].append(sum(errs) / len(errs))

        if evaluate_zmp:
            zmp_errors.append(evaluate_zmp_distance(pose_p, tran_p))

    if evaluate_pose:
        pose_errors = torch.stack(pose_errors).mean(dim=0)
        for name, error in zip(pose_evaluator.names, pose_errors):
            print('%s: %.4f' % (name, error[0]))
    if evaluate_zmp:
        print('ZMP Distance (m): %.4f' % (sum(zmp_errors) / len(zmp_errors)))
    if evaluate_tran:
        plt.plot([0] + [_ for _ in tran_errors.keys()], [0] + [torch.tensor(_).mean() for _ in tran_errors.values()], label=net.name)
        plt.legend(fontsize=15)
        plt.show()


if __name__ == '__main__':
    net = PIP()
    reduced_pose_evaluator = ReducedPoseEvaluator()
    full_pose_evaluator = FullPoseEvaluator()

    # Note: to evaluate Absolute Jitter Error, use full_pose_evaluator
    print('\n')
    evaluate(net, paths.totalcapture_dir, pose_evaluator=reduced_pose_evaluator, evaluate_pose=True, evaluate_tran=True, evaluate_zmp=True, flush_cache=False)

    print('\n')
    evaluate(net, paths.dipimu_dir, pose_evaluator=reduced_pose_evaluator, evaluate_pose=True, evaluate_zmp=True, flush_cache=False)
