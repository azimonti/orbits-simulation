import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from solar_system_body import Body
import sys
from scipy.integrate import solve_ivp

from types import SimpleNamespace

cfg = SimpleNamespace(
    # time_span=(0, 3.154e7),  # 1 year in seconds
    time_span=(0, 6.308e7),   # 2 year in seconds
    save_anim=False,          # save the animation
    animation_format='mp4',   # animation format (mp4 or gif)
    fps=30,                   # frame per second
    useRK45=True,             # use RK45
    verbose=True,             # verbose output
    use_yukawa=False,         # use Yukawa potential
    yukawa_coeff=1.5e15       # Yukawa potential coefficient
)


cst = SimpleNamespace(
    G=6.67430e-11,
    day_in_seconds=86400
)


def runge_kutta4(f, y0, t0, tf, dt, args=()):
    """
    RK4 integrator.

    Parameters:
    - f: Function to compute derivatives, should take (t, y, *args)
    - y0: Initial state vector.
    - t0: Initial time.
    - tf: Final time.
    - dt: Time step.
    - args: Additional arguments to pass to the function f.

    Returns:
    - t: Array of time points.
    - y: Array of state vectors at each time point.
    """
    num_steps = int(np.ceil((tf - t0) / dt)) + 1  # Including the initial point
    t = np.linspace(t0, tf, num_steps)
    y = np.zeros((len(y0), num_steps))
    y[:, 0] = y0

    for i in range(1, num_steps):
        ti = t[i - 1]
        yi = y[:, i - 1]
        k1 = dt * f(ti, yi, *args)
        k2 = dt * f(ti + dt / 2, yi + k1 / 2, *args)
        k3 = dt * f(ti + dt / 2, yi + k2 / 2, *args)
        k4 = dt * f(ti + dt, yi + k3, *args)
        y[:, i] = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, y


def gravitational_force_newton(m1, m2, r_ij):
    """Compute the gravitational force between two masses using Newtwon
    gravitational law."""
    r = np.linalg.norm(r_ij)
    return cst.G * m1 * m2 * r_ij / r**3


def gravitational_force_yukawa(m1, m2, r_ij):
    """Compute the gravitational force between two masses using Yukawa
    gravitational law."""
    r = np.linalg.norm(r_ij)
    return cst.G * m1 * m2 * r_ij / r**3 * (1 + r_ij/cfg.yukawa_coeff) * \
        np.exp(-r_ij / cfg.yukawa_coeff)


gravitational_force = gravitational_force_yukawa if cfg.use_yukawa else \
    gravitational_force_newton


def n_body_problem(t, y, masses):
    num_bodies = len(masses)
    positions = y[:3 * num_bodies].reshape((num_bodies, 3))
    velocities = y[3 * num_bodies:].reshape((num_bodies, 3))

    # initialize accelerations array
    accelerations = np.zeros_like(positions)

    # calculate accelerations due to gravity
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                # Vector from body i to body j
                r_ij = positions[j] - positions[i]
                # Add contribution of the force to the acceleration
                accelerations[i] += gravitational_force(
                    masses[i], masses[j], r_ij) / masses[i]

    # return the derivative of the state vector
    derivatives = np.concatenate(
        (velocities.flatten(), accelerations.flatten()))
    return derivatives


class SolarSystemSimulation:
    def __init__(self, outfile):
        self._bodies = []
        self._num_frames = 30
        self.outfile = outfile
        self.perc = None

    @property
    def bodies(self):
        return self._bodies

    def create_bodies(self):
        with open('data_input/bodies_2Dplane.json', 'r') as file:
            data_dict = json.load(file)
        for n in data_dict["bodies"]:
            b = json.loads(json.dumps(data_dict[n]),
                           object_hook=lambda d: SimpleNamespace(**d))
            self._bodies.append(
                Body(name=b.label, radius=b.radius,
                     mass=b.mass, scale=b.scale,  color=b.color,
                     position=b.position, velocity=b.velocity))

    def compute(self):
        t_eval = np.arange(0, cfg.time_span[1], cst.day_in_seconds)
        # extract masses
        masses = np.array([body.mass for body in self._bodies])
        # flatten initial conditions using the `position` #
        # and `velocity` methods from each body in the list `bodies`
        initial_conditions = np.concatenate(
            [np.concatenate([body.position for body in self._bodies]
                            + [body.velocity for body in self._bodies])])
        num_bodies = len(masses)
        if cfg.useRK45:
            # max_step is needed otherwise Rk45 is not coverging
            solution = solve_ivp(n_body_problem, cfg.time_span,
                                 initial_conditions, args=(masses,),
                                 method='RK45', t_eval=t_eval,
                                 max_step=10 * cst.day_in_seconds)
            self._positions = solution.y[:num_bodies * 3].reshape(
                (num_bodies, 3, -1))
            self._velocities = solution.y[num_bodies * 3:].reshape(
                (num_bodies, 3, -1))
        else:
            t, y = runge_kutta4(n_body_problem, initial_conditions,
                                cfg.time_span[0], cfg.time_span[1],
                                cst.day_in_seconds, args=(masses,))
            self._positions = y[:num_bodies * 3].reshape((num_bodies, 3, -1))
            self._velocities = y[num_bodies * 3:].reshape((num_bodies, 3, -1))

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

    def animate(self):
        self.perc = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.create_axes(ax)
        self._scat = []
        for index, body in enumerate(self._bodies):
            self._scat.append(
                ax.scatter([body.position[0]], [body.position[1]],
                           [body.position[2]],
                           color=body.color, s=body.mass_plot,
                           label=body.name, marker='o', zorder=index))

        distance = np.linalg.norm(self._bodies[1].position)
        distances = [np.linalg.norm(body.position) for body in self._bodies]
        distance = 1.1 * max(distances)

        ax.set_xlim(-distance, distance)
        ax.set_ylim(-distance, distance)
        ax.set_zlim(-distance, distance)
        ax.legend()

        def animate(frame):
            if cfg.verbose:
                perc = (frame + 1) / self._num_frames * 100
                if perc // 10 > self.perc // 10:
                    self.perc = perc
                    print(f"completed {int(perc)}% of the animation")
            for i in range(len(self._bodies)):
                # Extract the positions for body i at the current frame
                x, y, z = self._positions[i, :, frame]
                # Update the scatter plot offsets for body i
                self._scat[i]._offsets3d = ([x], [y], [z])
            return self._scat,

        # len(self._positions[0][1]) is the x component of the first body
        self._num_frames = len(self._positions[0][1])
        anim = FuncAnimation(
            fig, animate, frames=self._num_frames, interval=1000 / cfg.fps,
            blit=True)
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

    sim = SolarSystemSimulation(outfile)
    sim.create_bodies()
    sim.compute()
    sim.animate()


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
