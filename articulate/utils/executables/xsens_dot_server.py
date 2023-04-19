r"""
    Xsens Dot GUI server process. Run this server separately with your client to reduce sensor crashes.
"""


import torch
from queue import Empty
from articulate.utils.xsens import XsensDotSet
from articulate.utils.bullet.bullet import Button, Slider
from articulate.utils.bullet.view_rotation_np import RotationViewer
from pygame.time import Clock
import socket
import numpy as np


imus_addr = [
    'D4:22:CD:00:36:03',
    'D4:22:CD:00:44:6E',
    'D4:22:CD:00:45:E6',
    'D4:22:CD:00:45:EC',
    'D4:22:CD:00:46:0F',
    'D4:22:CD:00:32:32',
    # 'D4:22:CD:00:36:80',
    # 'D4:22:CD:00:36:04',
    # 'D4:22:CD:00:32:3E',
    # 'D4:22:CD:00:35:4E',
]

addr = ('127.0.0.1', 8777)
ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

XsensDotSet.set_buffer_len(2)
viewer = RotationViewer(n=len(imus_addr))
viewer.connect()
clock = Clock()

connect_btn = Button('connect', viewer.physics_client)
disconnect_btn = Button('disconnect', viewer.physics_client)
shutdown_btn = Button('shutdown', viewer.physics_client)
start_streaming_btn = Button('start streaming', viewer.physics_client)
stop_streaming_btn = Button('stop streaming', viewer.physics_client)
reset_heading_btn = Button('reset heading', viewer.physics_client)
revert_heading_btn = Button('revert heading', viewer.physics_client)
clear_btn = Button('clear', viewer.physics_client)
battery_btn = Button('battery info', viewer.physics_client)
quit_btn = Button('quit', viewer.physics_client)
fps_slider = Slider('fps', (1, 60), 60, viewer.physics_client)

while True:
    if connect_btn.is_click():
        XsensDotSet.sync_connect(imus_addr)
    if disconnect_btn.is_click():
        XsensDotSet.sync_disconnect()
    if shutdown_btn.is_click():
        XsensDotSet.sync_shutdown()
    if start_streaming_btn.is_click():
        XsensDotSet.start_streaming()
    if stop_streaming_btn.is_click():
        XsensDotSet.stop_streaming()
    if reset_heading_btn.is_click():
        XsensDotSet.reset_heading()
    if revert_heading_btn.is_click():
        XsensDotSet.revert_heading_to_default()
    if clear_btn.is_click():
        XsensDotSet.clear()
    if battery_btn.is_click():
        XsensDotSet.print_battery_info()
    if quit_btn.is_click():
        XsensDotSet.sync_disconnect()
        viewer.disconnect()
        break

    if XsensDotSet.is_started():
        clock.tick(fps_slider.get_int())
        try:
            T, Q, A = [], [], []
            for i in range(len(imus_addr)):
                t, q, a = XsensDotSet.get(i, timeout=1, preserve_last=True)
                T.append(t)
                Q.append(q)
                A.append(a)
                viewer.update(q.numpy()[[1, 2, 3, 0]], i)
            data = torch.cat((torch.tensor(T), torch.cat(Q), torch.cat(A)))
            data = data.numpy().astype(np.float32).tobytes()
            ss.sendto(data, addr)
            del data
        except Empty:
            print('[warning] read IMU error: Buffer for sensor %d is empty' % i)
