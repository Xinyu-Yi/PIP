r"""
    Calibration of RGB cameras.
    Press C to capture enough (>= max_frames) chessboard pictures.
"""

import os
import cv2
import numpy as np


max_frames = 30
wh = (1280, 720)
pattern_size = (9, 6)
save_dir = 'captured_pics/'

cv2.namedWindow('Camera View', 0)
cv2.namedWindow('Captured Image', 0)
cv2.imshow('Captured Image', np.zeros(wh + (3,)))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wh[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wh[1])

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= 38.6  # mm

obj_points = []
img_points = []
h, w = cap.read()[1].shape[:2]
used_frames = 0
print('Camera Resolution: [%d, %d]' % (w, h))
os.makedirs(save_dir, exist_ok=True)

while True:
    retval, img = cap.read()
    if not retval:
        break

    cv2.imshow('Camera View', img)
    if cv2.waitKey(1) == ord('c'):
        print(f'Searching for chessboard ... ', end='')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size, flags=cv2.CALIB_CB_FILTER_QUADS)
        if found:
            cv2.imwrite(os.path.join(save_dir, '%d.jpg' % used_frames), img)
            corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))
            used_frames += 1
            img_points.append(corners.reshape(1, -1, 2))
            obj_points.append(pattern_points.reshape(1, -1, 3))
            print('ok - %d/%d' % (used_frames, max_frames))
            if used_frames >= max_frames:
                print(f'Found {used_frames} frames with the chessboard.')
                break
        else:
            print('not found')

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.drawChessboardCorners(img, pattern_size, corners, found)
        cv2.imshow('Captured Image', img)

print('\ncalibrating...')
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
print("RMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

