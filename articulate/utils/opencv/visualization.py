r"""
    View 2D keypoints in real-time using opencv.
"""


__all__ = ['view_2d_keypoint', 'view_2d_keypoint_on_z_1']


import cv2
import numpy as np
import tqdm


def view_2d_keypoint(keypoints, parent=None, images=None, thickness=None, fps=60):
    r"""
    View 2d keypoint sequence in image coordinate frame. Modified from vctoolkit.render_bones_from_uv.

    Notes
    -----
    If num_frame == 1, only show one picture.
    If parent is None, do not render bones.
    If images is None, use 1080p white canvas.
    If thickness is None, use a default value.
    If keypoints in shape [..., 2], render keypoints without confidence.
    If keypoints in shape [..., 3], render confidence using alpha of colors (more transparent, less confident).

    :param keypoints: Tensor [num_frames, num_joints, *] where *=2 for (u, v) and *=3 for (u, v, confidence).
    :param parent: List in length [num_joints]. e.g., [None, 0, 0, 0, 1, 2, 3 ...]
    :param images: Numpy uint8 array that can expand to [num_frame, height, width, 3].
    :param thickness: Thickness for points and lines.
    :param fps: Sequence FPS.
    """
    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
    if images is None:
        images = np.ones((keypoints.shape[0], 540, 960, 3), dtype=np.uint8) * 255
    if images.dtype != np.uint8:
        raise RuntimeError('images must be uint8 type')
    if thickness is None:
        thickness = int(max(round(images.shape[1] / 160), 1))
    images = np.broadcast_to(images, (keypoints.shape[0], images.shape[-3], images.shape[-2], 3))
    has_conf = keypoints.shape[-1] == 3
    is_single_frame = len(images) == 1

    if not is_single_frame:
        writer = cv2.VideoWriter('a.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (images.shape[2], images.shape[1]))
    for i in tqdm.trange(len(images)):
        bg = images[i]
        for uv in keypoints[i]:
            conf = float(uv[2]) if has_conf else 1
            fg = cv2.circle(bg.copy(), (int(uv[0]), int(uv[1])), int(thickness * 2), (0, 0, 255), -1)
            bg = cv2.addWeighted(bg, 1 - conf, fg, conf, 0)
        if parent is not None:
            for c, p in enumerate(parent):
                if p is not None:
                    start = (int(keypoints[i][p][0]), int(keypoints[i][p][1]))
                    end = (int(keypoints[i][c][0]), int(keypoints[i][c][1]))
                    conf = min(float(keypoints[i][c][2]), float(keypoints[i][p][2])) if has_conf else 1
                    fg = cv2.line(bg.copy(), start, end, (255, 0, 0), thickness)
                    bg = cv2.addWeighted(bg, 1 - conf, fg, conf, 0)
        cv2.imshow('2d keypoint', bg)
        if is_single_frame:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
            writer.write(bg)
    if not is_single_frame:
        writer.release()
    cv2.destroyWindow('2d keypoint')


def view_2d_keypoint_on_z_1(keypoints, parent=None, thickness=None, scale=1, fps=60):
    r"""
    View 2d keypoint sequence on z=1 plane.

    Notes
    -----
    If num_frame == 1, only show one picture.
    If parent is None, do not render bones.
    If thickness is None, use a default value.
    If keypoints in shape [..., 2], render keypoints without confidence.
    If keypoints in shape [..., 3], render confidence using alpha of colors (more transparent, less confident).

    :param keypoints: Tensor [num_frames, num_joints, *] where *=2 for (x, y) and *=3 for (x, y, confidence).
    :param parent: List in length [num_joints]. e.g., [None, 0, 0, 0, 1, 2, 3 ...]
    :param thickness: Thickness for points and lines.
    :param scale: Scale of the keypoints.
    :param fps: Sequence FPS.
    """
    f = 500 * scale
    keypoints = keypoints.clone()
    keypoints[..., 0] = keypoints[..., 0] * f + 960 / 2
    keypoints[..., 1] = keypoints[..., 1] * f + 540 / 2
    view_2d_keypoint(keypoints, parent=parent, thickness=thickness, fps=fps)
