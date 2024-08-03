import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import sys
from types import SimpleNamespace


class Body:
    def __init__(self, name='', radius=0, mass=0,
                 position=np.zeros(3), velocity=np.zeros(3)):
        self._name = name
        self._radius = radius
        self._mass = mass
        self._position = position
        self._velocity = velocity

    @property
    def name(self):
        return self._name

    @property
    def radius(self):
        return self._radius

    @property
    def mass(self):
        return self._mass

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = np.array(velocity)


def make_plot():
    with open('bodies.json', 'r') as file:
        data_dict = json.load(file)

    bodies = []

    # create Sun and Earth bodies
    for n in ['sun', 'earth']:
        b = json.loads(json.dumps(data_dict[n]),
                       object_hook=lambda d: SimpleNamespace(**d))

        bodies.append(Body(name=b.label, radius=b.radius, mass=b.mass,
                           position=b.position, velocity=b.velocity))

    # Visualization
    fig = plt.figure()
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} " \
        r"\usepackage{amsmath} \usepackage{helvet}"
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica"
    })
    ax = fig.add_subplot(111, projection='3d')
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
    # define a function to format the tick labels blank so tight_layout
    # can be applied

    def format_func(value, tick_number):
        return ' '  # single space
    # apply the formatter to the x, y, and z axes
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    ax.zaxis.set_major_formatter(FuncFormatter(format_func))
    plt.tight_layout()

    # plot Sun
    sun = bodies[0]
    ax.scatter(sun.position[0], sun.position[1], sun.position[2],
               color='yellow', s=np.log(sun.mass),
               label=sun.name, marker='o')

    # plot Earth
    earth = bodies[1]
    ax.scatter(earth.position[0], earth.position[1], earth.position[2],
               color='blue', s=np.log(earth.mass),
               label=earth.name, marker='o')

    distance = np.linalg.norm(earth.position)
    ax.set_xlim(-distance, distance)
    ax.set_ylim(-distance, distance)
    ax.set_zlim(-distance, distance)
    # Labels and legend
    ax.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='solar system simulation')
    args = parser.parse_args()
    if args:
        ...
    make_plot()


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
