r"""
    Client codes for xsens dots. Run the server `articulate/utils/executables/xsens_dot_server.py` first.
"""


import numpy as np
import time
import socket
import torch
from pygame.time import Clock
from net import PIP
import articulate as art
import win32api
import os
from config import *
import keyboard
import datetime


class IMUSet:
    def __init__(self):
        self.n_imus = 6
        self.cs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cs.bind(('127.0.0.1', 8777))
        self.cs.settimeout(3)

    def get(self):
        try:
            data, server = self.cs.recvfrom(32 * self.n_imus)
            data = np.frombuffer(data, np.float32)
            t = torch.tensor(data[:self.n_imus])
            q = torch.tensor(data[self.n_imus:5 * self.n_imus]).view(self.n_imus, 4)
            a = torch.tensor(data[5 * self.n_imus:]).view(self.n_imus, 3)
            return t, q, a
        except socket.timeout:
            print('[warning] no imu data received for 3 seconds')

    def clear(self):
        while True:
            t1 = time.time()
            self.cs.recvfrom(int(32 * 6))
            self.cs.recvfrom(int(32 * 6))
            self.cs.recvfrom(int(32 * 6))
            t2 = time.time()
            if t2 - t1 > 2.5 / 60:
                break


def tpose_calibration():
    c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Forward, y = Left, z = Up).')
    if c == 'n' or c == 'N':
        imu_set.clear()
        RSI = art.math.quaternion_to_rotation_matrix(imu_set.get()[1][0]).view(3, 3).t()
        RMI = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0.]]).mm(RSI)
        torch.save(RMI, os.path.join(paths.temp_dir, 'RMI.pt'))
    else:
        RMI = torch.load(os.path.join(paths.temp_dir, 'RMI.pt'))
    print(RMI)

    input('Stand straight in T-pose and press enter. The calibration will begin in 3 seconds')
    time.sleep(3)
    imu_set.clear()
    RIS = art.math.quaternion_to_rotation_matrix(imu_set.get()[1])
    RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))  # = (R_MI R_IS)^T R_MB = R_SB
    return RMI, RSB


if __name__ == '__main__':
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)

    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    if False and os.path.exists(paths.unity_file):
        win32api.ShellExecute(0, 'open', os.path.abspath(paths.unity_file), '', '', 1)
    conn, addr = server_for_unity.accept()

    imu_set = IMUSet()
    RMI, RSB = tpose_calibration()
    net = PIP()
    clock = Clock()
    imu_set.clear()
    data = {'RMI': RMI, 'RSB': RSB, 'aM': [], 'RMB': []}

    while True:
        clock.tick(63)
        tframe, q, a = imu_set.get()
        RMB = RMI.matmul(art.math.quaternion_to_rotation_matrix(q)).matmul(RSB)
        aM = a.mm(RMI.t())
        pose, tran = net.forward_frame(aM.view(1, 6, 3), RMB.view(1, 6, 3, 3))
        pose = art.math.rotation_matrix_to_axis_angle(pose).view(-1, 72)
        tran = tran.view(-1, 3)

        # send pose
        s = ','.join(['%g' % v for v in pose.view(-1)]) + '#' + \
            ','.join(['%g' % v for v in tran.view(-1)]) + '$'
        conn.send(s.encode('utf8'))

        data['aM'].append(aM)
        data['RMB'].append(RMB)

        if keyboard.is_pressed('q'):
            break

        print('\rfps: ', clock.get_fps(), end='')

    if os.path.exists(paths.unity_file):
        os.system('taskkill /F /IM "%s"' % os.path.basename(paths.unity_file))

    data['aM'] = torch.stack(data['aM'])
    data['RMB'] = torch.stack(data['RMB'])
    torch.save(data, os.path.join(paths.live_record_dir, 'xsens' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pt'))
    print('Finish.')
