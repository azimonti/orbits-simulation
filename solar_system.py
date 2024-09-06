#!/usr/bin/env python3
'''
/**********************/
/*  solar_system.py   */
/*    Version 1.0     */
/*     2024/08/03     */
/**********************/
'''
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
    save_anim=True,           # save the animation
    animation_format='mp4',   # animation format (mp4 or gif)
    fps=30,                   # frame per second
    useRK45=True,             # use RK45
    step=86400,               # integration step 1 day
    verbose=True,             # verbose output
    use_yukawa=False,         # use Yukawa potential
    yukawa_coeff=1.5e15,      # Yukawa potential coefficient
    plot_orbits=True,         # plot the orbits
    project_orbits_2d=True,   # plot the orbits on the bottom
    project_2d=True,          # project on the bottom of the grid
    high_res_plot=True,       # enable high resolution simulation plot
    fig_4k=True               # use 4k resolution
)


cst = SimpleNamespace(
    G=6.67430e-11
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
    """Compute the gravitational force between two masses using Newton
    gravitational law."""
    r = np.linalg.norm(r_ij)
    return cst.G * m1 * m2 * r_ij / r**3


def gravitational_force_yukawa(m1, m2, r_ij):
    """Compute the gravitational force between two masses using Yukawa
    gravitational law."""
    r = np.linalg.norm(r_ij)
    return cst.G * m1 * m2 * r_ij / r**3 * (1 + r / cfg.yukawa_coeff) * \
        np.exp(-r / cfg.yukawa_coeff)


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
        self._outfile = outfile
        self._perc = None

    @property
    def bodies(self):
        return self._bodies

    def create_bodies(self):
        with open('data_input/bodies_3D.json', 'r') as file:
            data_dict = json.load(file)
        for n in data_dict["bodies"]:
            b = json.loads(json.dumps(data_dict[n]),
                           object_hook=lambda d: SimpleNamespace(**d))
            self._bodies.append(
                Body(name=b.label, radius=b.radius,
                     mass=b.mass, scale=b.scale, color=b.color,
                     position=b.position, velocity=b.velocity))

    def compute(self):
        t_eval = np.arange(0, cfg.time_span[1], cfg.step)
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
                                 max_step=cfg.step)
            self._positions = solution.y[:num_bodies * 3].reshape(
                (num_bodies, 3, -1))
            self._velocities = solution.y[num_bodies * 3:].reshape(
                (num_bodies, 3, -1))
        else:
            t, y = runge_kutta4(n_body_problem, initial_conditions,
                                cfg.time_span[0], cfg.time_span[1],
                                cfg.step, args=(masses,))
            self._positions = y[:num_bodies * 3].reshape((num_bodies, 3, -1))
            self._velocities = y[num_bodies * 3:].reshape((num_bodies, 3, -1))

    def create_axes(self, ax, scale):
        ax.grid(True)
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.rcParams['lines.dashed_pattern'] = [20, 20]
        ax.xaxis._axinfo['grid'].update(
            {"linewidth": scale * 0.5, "color": (0, 0, 0), 'linestyle': '--'})
        ax.yaxis._axinfo['grid'].update(
            {"linewidth": scale * 0.5, "color": (0, 0, 0), 'linestyle': '--'})
        ax.zaxis._axinfo['grid'].update(
            {"linewidth": scale * 0.5, "color": (0, 0, 0), 'linestyle': '--'})

        ax.xaxis.line.set_color((0.0, 0.0, 0.0, 1.0))
        ax.xaxis.line.set_linewidth(scale * 1.0)
        ax.yaxis.line.set_color((0.0, 0.0, 0.0, 1.0))
        ax.yaxis.line.set_linewidth(scale * 1.0)
        ax.zaxis.line.set_color((0.0, 0.0, 0.0, 1.0))
        ax.zaxis.line.set_linewidth(scale * 1.0)
        ax.set_xlabel('', fontsize=10)
        ax.set_ylabel('', fontsize=10)
        ax.set_zlabel('', fontsize=10)
        ax.tick_params(axis='x', color='w')
        ax.tick_params(axis='y', color='w')
        ax.tick_params(axis='z', color='w')

        def format_func(value, tick_number):
            # define a function to format the tick labels blank so tight_layout
            # can be applied
            return ' '  # single space
        # apply the formatter to the x, y, and z axes
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        ax.zaxis.set_major_formatter(FuncFormatter(format_func))
        ax.tick_params(axis='x', which='both', width=0.1, length=5)
        ax.tick_params(axis='y', which='both', width=0.1, length=5)
        ax.tick_params(axis='z', which='both', width=0.1, length=5)
        ax.xaxis.set_tick_params(width=5)
        ax.yaxis.set_tick_params(width=5)

        plt.tight_layout()

    def animate(self):
        self._perc = 0
        dpi = 300 if (cfg.high_res_plot and not cfg.save_anim) else 100
        if cfg.fig_4k:
            figsize = (3840 / dpi, 2160 / dpi)
            scale1 = 4.0
            scale2 = 20.0
        else:
            figsize = (1920 / dpi, 1080 / dpi)
            scale1 = 1.0
            scale2 = 1.0
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        self.create_axes(ax, scale1)
        self._scat = []
        self._scat2 = []
        distances = [np.linalg.norm(body.position) for body in self._bodies]
        self.axs_limit = 1.1 * max(distances)
        for index, body in enumerate(self._bodies):
            self._scat.append(
                ax.scatter([body.position[0]], [body.position[1]],
                           [body.position[2]],
                           color=body.color, s=scale2 * body.mass_plot,
                           label=body.name, marker='o', zorder=index))
            if cfg.project_2d:
                self._scat2.append(
                    ax.scatter([body.position[0]], [body.position[1]],
                               -self.axs_limit,
                               color=(.5, .5, .5), s=scale2 * body.mass_plot,
                               label=body.name, marker='o', zorder=index))
            if cfg.plot_orbits:
                x_positions = self._positions[index, 0, :]
                y_positions = self._positions[index, 1, :]
                if cfg.project_orbits_2d:
                    z_positions = [-self.axs_limit] * len(x_positions)
                else:
                    z_positions = self._positions[index, 2, :]
                ax.plot(
                    x_positions, y_positions, z_positions,
                    color=(.5, .5, .5), linewidth=scale1 * 0.8)
        ax.set_xlim(-self.axs_limit, self.axs_limit)
        ax.set_ylim(-self.axs_limit, self.axs_limit)
        ax.set_zlim(-self.axs_limit, self.axs_limit)
        scat_legend_handles = [scatter for scatter in self._scat]
        ax.legend(handles=scat_legend_handles, fontsize=scale1 * 20,
                  loc='upper right', bbox_to_anchor=(1.2, 1), frameon=False)

        def animate_frame(frame):
            if cfg.verbose:
                perc = (frame + 1) / self._num_frames * 100
                if perc // 10 > self._perc // 10:
                    self._perc = perc
                    print(f"completed {int(perc)}% of the animation")
            for i in range(len(self._bodies)):
                # Extract the positions for body i at the current frame
                x, y, z = self._positions[i, :, frame]
                # Update the scatter plot offsets for body i
                self._scat[i]._offsets3d = ([x], [y], [z])
                if cfg.project_2d:
                    self._scat2[i]._offsets3d = (
                        [x], [y], [-self.axs_limit])
            return self._scat, self._scat2

        # len(self._positions[0][1]) is the x component of the first body
        self._num_frames = len(self._positions[0][1])
        anim = FuncAnimation(
            fig, animate_frame, frames=self._num_frames,
            interval=1000 / cfg.fps, blit=True)
        if cfg.save_anim:
            base, ext = self._outfile.rsplit('.', 1)
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
