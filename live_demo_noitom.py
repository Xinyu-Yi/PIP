r"""
    Live demo using Noitom Perception Neuron Lab IMUs.
"""


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
from articulate.utils.noitom import *


class IMUSet:
    g = 9.8

    def __init__(self, udp_port=7777):
        app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_calc_data()
        app.set_settings(settings)
        app.open()
        time.sleep(0.5)

        sensors = [None for _ in range(6)]
        evts = []
        while len(evts) == 0:
            evts = app.poll_next_event()
            for evt in evts:
                assert evt.event_type == MCPEventType.SensorModulesUpdated
                sensor_module_handle = evt.event_data.sensor_module_data.sensor_module_handle
                sensor_module = MCPSensorModule(sensor_module_handle)
                sensors[sensor_module.get_id() - 1] = sensor_module

        print('find %d sensors' % len([_ for _ in sensors if _ is not None]))
        self.app = app
        self.sensors = sensors
        self.t = 0

    def get(self):
        evts = self.app.poll_next_event()
        if len(evts) > 0:
            self.t = evts[0].timestamp
        q, a = [], []
        for sensor in self.sensors:
            q.append(sensor.get_posture())
            a.append(sensor.get_accelerated_velocity())

        # assuming g is positive (= 9.8), we need to change left-handed system to right-handed by reversing axis x, y, z
        R = art.math.quaternion_to_rotation_matrix(torch.tensor(q))  # rotation is not changed
        a = -torch.tensor(a) / 1000 * self.g                         # acceleration is reversed
        a = R.bmm(a.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])   # calculate global free acceleration
        return self.t, R, a

    def clear(self):
        pass


def tpose_calibration():
    c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Right, y = Forward, z = Down, Left-handed).')
    if c == 'n' or c == 'N':
        imu_set.clear()
        RSI = imu_set.get()[1][0].view(3, 3).t()
        RMI = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI)
        torch.save(RMI, os.path.join(paths.temp_dir, 'RMI.pt'))
    else:
        RMI = torch.load(os.path.join(paths.temp_dir, 'RMI.pt'))
    print(RMI)

    input('Stand straight in T-pose and press enter. The calibration will begin in 3 seconds')
    time.sleep(3)
    imu_set.clear()
    RIS = imu_set.get()[1]
    RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))
    return RMI, RSB


def test_sensors():
    from articulate.utils.bullet import RotationViewer
    from articulate.utils.pygame import StreamingDataViewer
    clock = Clock()
    imu_set = IMUSet()
    with RotationViewer(6) as rviewer, StreamingDataViewer(6, (-10, 10)) as sviewer:
        imu_set.clear()
        while True:
            clock.tick(60)
            t, R, a = imu_set.get()
            rviewer.update_all(R)
            sviewer.plot(a[:, 1])
            print('time: %.3f' % t, '\tacc:', a.norm(dim=1))


if __name__ == '__main__':
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)

    is_executable = False
    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    if paths.unity_file != '' and os.path.exists(paths.unity_file):
        win32api.ShellExecute(0, 'open', os.path.abspath(paths.unity_file), '', '', 1)
        is_executable = True
    conn, addr = server_for_unity.accept()

    imu_set = IMUSet()
    RMI, RSB = tpose_calibration()
    net = PIP()
    clock = Clock()
    imu_set.clear()
    data = {'RMI': RMI, 'RSB': RSB, 'aM': [], 'RMB': []}

    while True:
        clock.tick(60)
        tframe, RIS, aI = imu_set.get()
        RMB = RMI.matmul(RIS).matmul(RSB)
        aM = aI.mm(RMI.t())
        pose, tran, cj, grf = net.forward_frame(aM.view(1, 6, 3), RMB.view(1, 6, 3, 3), return_grf=True)
        pose = art.math.rotation_matrix_to_axis_angle(pose).view(-1, 72)
        tran = tran.view(-1, 3)

        # send motion to Unity
        s = ','.join(['%g' % v for v in pose.view(-1)]) + '#' + \
            ','.join(['%g' % v for v in tran.view(-1)]) + '#' + \
            ','.join(['%d' % v for v in cj]) + '#' + \
            (','.join(['%g' % v for v in grf.view(-1)]) if grf is not None else '') + '$'

        try:
            conn.send(s.encode('utf8'))
        except:
            break

        data['aM'].append(aM)
        data['RMB'].append(RMB)

        if is_executable and keyboard.is_pressed('q'):
            break

        print('\rfps: ', clock.get_fps(), end='')

    if is_executable:
        os.system('taskkill /F /IM "%s"' % os.path.basename(paths.unity_file))

    data['aM'] = torch.stack(data['aM'])
    data['RMB'] = torch.stack(data['RMB'])
    torch.save(data, os.path.join(paths.live_record_dir, 'xsens' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pt'))
    print('\rFinish.')
