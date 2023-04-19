r"""
    Xsens Dot cmd server process. Run this server separately with your client to reduce sensor crashes.
"""
import time

import torch
from queue import Empty
from articulate.utils.xsens import XsensDotSet
from pygame.time import Clock
import socket
import numpy as np
import keyboard
import articulate as art
from articulate.utils.print import *


imus_addr = [
    # 'D4:22:CD:00:36:03',
    'D4:22:CD:00:44:6E',
    'D4:22:CD:00:45:E6',
    'D4:22:CD:00:45:EC',
    # 'D4:22:CD:00:46:0F',
    'D4:22:CD:00:32:32',
    # 'D4:22:CD:00:36:80',
    # 'D4:22:CD:00:36:04',
    # 'D4:22:CD:00:32:3E',
    # 'D4:22:CD:00:35:4E',
]

addr = ('127.0.0.1', 8777)
ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
XsensDotSet.set_buffer_len(2)
clock = Clock()
is_hotkey_locked = True
fps = 60

helps = r'''
================= Hotkey =================
You need to unlock the hotkeys first.

l / shift + l       lock / unlock all hotkeys
h                   print this help
c / shift + c       connect / disconnect
s / shift + s       start / stop streaming
a / shift + a       reset / revert heading
r                   clear buffer
b                   print battery info
o                   power off
d                   print sensor angle
esc                 disconnect and quit
1 ~ 6               set fps to 10 ~ 60 (default 60)
'''
print(helps)

while True:
    if is_hotkey_locked:
        if keyboard.is_pressed('shift+l'):
            is_hotkey_locked = False
            print_green('hotkey unlocked')
            time.sleep(0.5)
    else:
        if keyboard.is_pressed('l'):
            is_hotkey_locked = True
            print_green('hotkey locked')
            time.sleep(0.5)
        if keyboard.is_pressed('h'):
            print(helps)
            time.sleep(0.5)
        if keyboard.is_pressed('shift+c'):
            XsensDotSet.sync_disconnect()
            print_green('sensor disconnected')
        elif keyboard.is_pressed('c'):
            XsensDotSet.sync_connect(imus_addr)
            print_green('sensor connected')
        if keyboard.is_pressed('o'):
            XsensDotSet.sync_shutdown()
            print_green('sensor powered off')
        if keyboard.is_pressed('shift+s'):
            XsensDotSet.stop_streaming()
            print_green('streaming stopped')
        elif keyboard.is_pressed('s'):
            XsensDotSet.start_streaming()
            print_green('streaming started')
        if keyboard.is_pressed('shift+a'):
            XsensDotSet.revert_heading_to_default()
            print_green('heading reverted')
        elif keyboard.is_pressed('a'):
            XsensDotSet.reset_heading()
            print_green('heading reset')
        if keyboard.is_pressed('r'):
            XsensDotSet.clear()
            print_green('buffer cleared')
        if keyboard.is_pressed('b'):
            XsensDotSet.print_battery_info()
        if keyboard.is_pressed('esc'):
            XsensDotSet.sync_disconnect()
            break
        if keyboard.is_pressed('1'):
            fps = 10
            print_green('fps set to 10')
            time.sleep(0.5)
        if keyboard.is_pressed('2'):
            fps = 20
            print_green('fps set to 20')
            time.sleep(0.5)
        if keyboard.is_pressed('3'):
            fps = 30
            print_green('fps set to 30')
            time.sleep(0.5)
        if keyboard.is_pressed('4'):
            fps = 40
            print_green('fps set to 40')
            time.sleep(0.5)
        if keyboard.is_pressed('5'):
            fps = 50
            print_green('fps set to 50')
            time.sleep(0.5)
        if keyboard.is_pressed('6'):
            fps = 60
            print_green('fps set to 60')
            time.sleep(0.5)

    clock.tick(fps)
    if XsensDotSet.is_started():
        try:
            T, Q, A = [], [], []
            for i in range(len(imus_addr)):
                t, q, a = XsensDotSet.get(i, timeout=1, preserve_last=True)
                T.append(t)
                Q.append(q)
                A.append(a)
            if not is_hotkey_locked and keyboard.is_pressed('d'):
                dq = ['%.1f' % art.math.radian_to_degree(art.math.angle_between(Q[0], q, art.math.RotationRepresentation.QUATERNION)) for q in Q]
                print('angle (deg):', dq)
            data = torch.cat((torch.tensor(T), torch.cat(Q), torch.cat(A)))
            data = data.numpy().astype(np.float32).tobytes()
            ss.sendto(data, addr)
            del data
        except Empty:
            print('[warning] read IMU error: Buffer for sensor %d is empty' % i)
