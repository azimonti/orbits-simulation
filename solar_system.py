import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from solar_system_body import Body
import sys
from types import SimpleNamespace

cfg = SimpleNamespace(
    save_anim=False,          # save the animation
    animation_format='mp4',   # animation format (mp4 or gif)
    fps=30,                   # frame per second
    verbose=True
)


class SolarSystem:
    def __init__(self, outfile):
        self._bodies = []
        self._num_frames = 30
        self.outfile = outfile
        self.perc = None

    @property
    def bodies(self):
        return self._bodies

    def create_bodies(self):
        with open('bodies.json', 'r') as file:
            data_dict = json.load(file)
        for n in ['sun', 'earth']:
            b = json.loads(json.dumps(data_dict[n]),
                           object_hook=lambda d: SimpleNamespace(**d))
            self._bodies.append(
                Body(name=b.label, radius=b.radius,
                     mass=b.mass, scale=b.scale,
                     position=b.position, velocity=b.velocity))

    def create_axes(self, ax):
        ax.grid(True)
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.rcParams['lines.dashed_pattern'] = [20, 20]
        ax.xaxis._axinfo['grid'].update(
            {"linewidth": 0.1, "color": (0, 0, 0), 'linestyle': '--'})
        ax.yaxis._axinfo['grid'].update(
            {"linewidth": 0.1, "color": (0, 0, 0), 'linestyle': '--'})
        ax.zaxis._axinfo['grid'].update(
            {"linewidth": 0.1, "color": (0, 0, 0), 'linestyle': '--'})

        ax.xaxis.line.set_color((0.65, 0.65, 0.65, 1.0))
        ax.yaxis.line.set_color((0.65, 0.65, 0.65, 1.0))
        ax.zaxis.line.set_color((0.65, 0.65, 0.65, 1.0))
        ax.set_xlabel('', fontsize=10)
        ax.set_ylabel('', fontsize=10)
        ax.set_zlabel('', fontsize=10)

        def format_func(value, tick_number):
            # define a function to format the tick labels blank so tight_layout
            # can be applied
            return ' '  # single space
        # apply the formatter to the x, y, and z axes
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        ax.zaxis.set_major_formatter(FuncFormatter(format_func))
        plt.tight_layout()

    def plot_simulation(self):
        self.perc = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.create_axes(ax)
        # plot Sun
        self._scat = []
        sun = self._bodies[0]
        self._scat.append(
            ax.scatter([sun.position[0]], [sun.position[1]], [sun.position[2]],
                       color='yellow', s=sun.mass_plot,
                       label=sun.name, marker='o', zorder=1))

        # plot Earth
        earth = self._bodies[1]
        self._scat.append(
            ax.scatter([earth.position[0]], [earth.position[1]],
                       [earth.position[2]], color='blue', s=earth.mass_plot,
                       label=earth.name, marker='o', zorder=2))

        # self.graph = ax.scatter(0,0,0)
        distance = np.linalg.norm(earth.position)
        ax.set_xlim(-distance, distance)
        ax.set_ylim(-distance, distance)
        ax.set_zlim(-distance, distance)
        # labels and legend
        ax.legend()

        def animate(frame):
            if cfg.verbose:
                perc = (frame + 1) / self._num_frames * 100
                if perc // 10 > self.perc // 10:
                    self.perc = perc
                    print(f"completed {int(perc)}% of the animation")
            self._scat[0]._offsets3d = ([frame * 1e10], [0], [0])
            return self._scat,

        anim = FuncAnimation(
            fig, animate, frames=self._num_frames, interval=1000 / cfg.fps,
            blit=False)
        if cfg.save_anim:
            base, ext = self.outfile.rsplit('.', 1)
            animation_format = cfg.animation_format
            outfile_a = f"{base}.{animation_format}"
            if animation_format == 'mp4':
                anim.save(outfile_a, writer='ffmpeg')
            elif animation_format == 'gif':
                anim.save(outfile_a, writer='imagemagick')
        else:
            plt.show()


def make_plot(outfile: str):
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} " \
        r"\usepackage{amsmath} \usepackage{helvet}"
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica"
    })
    plt.rcParams['animation.convert_path'] = 'magick'

    s = SolarSystem(outfile)
    s.create_bodies()
    s.plot_simulation()


def main():
    parser = argparse.ArgumentParser(
        description='solar system simulation')
    parser.add_argument('-o', '--ofile', help='output file')
    args = parser.parse_args()
    if args.ofile:
        ofile = args.ofile
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        ofile = tmp_dir + "/solar_system.png"
    make_plot(ofile)


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
