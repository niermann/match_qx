import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider


class DraggableLineScan:
    lock = None  # only one can be animated at a time

    def __init__(self, image, pos0, pos1, width, title=None):
        self.pos = [np.array(pos0, dtype=float), np.array(pos1, dtype=float)]
        self.width = float(width)
        self.points = [patches.Circle(self.pos[0], 3, color='r', linewidth=0),
                       patches.Circle(self.pos[1], 3, color='r', linewidth=0)]
        self.title = title

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.ax.imshow(image, cmap="gray")

        ax_width = plt.axes((0.25, 0.1, 0.65, 0.03))
        self.width_slider = Slider(ax=ax_width, label="Width", valmin=0, valmax=300, valinit=self.width)

        for p in self.points:
            self.ax.add_patch(p)

        self.press = None
        self.patch = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.width_slider.on_changed(self.on_width)
        self.update()

    def on_width(self, value):
        self.width = value
        self.update()

    def update(self):
        ppdir = np.array((self.pos[0][1] - self.pos[1][1], self.pos[1][0] - self.pos[0][0]), dtype=float)
        ppdir /= np.sqrt(np.sum(ppdir ** 2))

        v = [self.pos[0] + self.width * 0.5 * ppdir,
             self.pos[0] - self.width * 0.5 * ppdir,
             self.pos[1] - self.width * 0.5 * ppdir,
             self.pos[1] + self.width * 0.5 * ppdir,
             self.pos[0] + self.width * 0.5 * ppdir]

        if self.patch:
            self.patch.remove()

        delta = self.pos[1] - self.pos[0]
        self.patch = self.ax.add_patch(patches.Polygon(v, fill=False, edgecolor="white"))
        title = (self.title + '\n') if self.title else ''
        self.ax.set_title(title + f'Length: {np.sqrt(np.sum(delta ** 2)):.1f} px,' +
                          f' Angle: {np.degrees(np.arctan2(delta[1], delta[0])):.1f} deg')
        plt.draw()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if self.lock is None:
            return

        point = self.lock
        for n, p in enumerate(self.points):
            if p is point:
                pt_idx = n
                break
        else:
            return

        pos = np.array((event.xdata, event.ydata), dtype=float)
        point.center = pos
        self.pos[pt_idx] = pos

        self.ax.draw_artist(point)
        self.update()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.lock is not None:
            return

        for point in self.points[::-1]:
            if point.contains(event)[0]:
                break
        else:
            return

        self.lock = point
        point.set_color('yellow')

        self.ax.draw_artist(point)
        plt.draw()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        if self.lock is None:
            return

        point = self.lock
        point.set_color('r')
        self.lock = None

        self.ax.draw_artist(point)
        self.update()

    def run(self):
        plt.show()
