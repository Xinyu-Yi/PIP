r"""
    View streaming 1D data in real-time using pygame.
"""


__all__ = ['StreamingDataViewer']


import pygame
import numpy as np
from collections import deque
import matplotlib


class StreamingDataViewer:
    r"""
    View 1D streaming data in real-time.
    """
    W = 800
    H = 600
    colors = (np.array(matplotlib.colormaps['tab10'].colors) * 255).astype(int)

    def __init__(self, n=1, y_range=(0, 10), max_history_length=100):
        r"""
        :param n: Number of data to simultaneously plot.
        :param y_range: Data range (min, max).
        :param max_history_length: Max number of historical data points simultaneously shown in screen for each plot.
        """
        self.n = n
        self.max_history_length = max_history_length
        self.y_range = y_range
        self.ys = None
        self.screen = None
        self.dx = self.W / (max_history_length - 1)
        self.dy = self.H / (y_range[1] - y_range[0])
        self.line_width = max(self.H // 200, 1)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        pygame.init()
        self.ys = [deque(maxlen=self.max_history_length) for _ in range(self.n)]
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption('Streaming Data Viewer: x_length=%d, y_range=(%.1f, %.1f)' %
                                   (self.max_history_length, self.y_range[0], self.y_range[1]))

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if self.screen is not None:
            self.screen = None
            pygame.quit()

    def plot(self, values):
        r"""
        Plot all current values.

        :param values: Iterable in length n.
        """
        if self.screen is None:
            print('[Error] StreamingDataViewer is not connected.')
            return
        assert len(values) == self.n, 'Number of data is not equal to the init value in StreamingDataViewer.'
        self.screen.fill((255, 255, 255))
        for i, v in enumerate(values):
            self.ys[i].append(float(v))
            points = [(j * self.dx, self.H - (v - self.y_range[0]) * self.dy) for j, v in enumerate(self.ys[i])]
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors[i % 10], False, points, width=self.line_width)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.disconnect()


# example
if __name__ == '__main__':
    import time
    with StreamingDataViewer(3, y_range=(0, 2), max_history_length=100) as viewer:
        for _ in range(300):
            viewer.plot(np.random.rand(3))
            time.sleep(1 / 30)
