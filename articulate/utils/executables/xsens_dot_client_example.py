r"""
    An example of Xsens Dot client.
"""

import time
import socket
import numpy as np
from pygame.time import Clock

n_imus = 6   # must be the same as the server
cs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cs.bind(('127.0.0.1', 8777))
cs.settimeout(5)
clock = Clock()

# clear the udp buffer
while True:
    t1 = time.time()
    cs.recvfrom(int(32 * n_imus))
    cs.recvfrom(int(32 * n_imus))
    cs.recvfrom(int(32 * n_imus))
    t2 = time.time()
    if t2 - t1 > 2.5 / 60:
        break

while True:
    try:
        clock.tick()
        data, server = cs.recvfrom(32 * n_imus)
        data = np.frombuffer(data, np.float32)
        t = data[:n_imus]
        q = data[n_imus:5 * n_imus].reshape(n_imus, 4)
        a = data[5 * n_imus:].reshape(n_imus, 3)
        print('\r', clock.get_fps(), end='')
        # print(server, t, q, a)
    except socket.timeout:
        print('[warning] no data received for 5 seconds')
