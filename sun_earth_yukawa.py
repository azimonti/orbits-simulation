#!/usr/bin/env python3
'''
/***************************/
/*   sun_earth_yukawa.py   */
/*      Version 1.0        */
/*       2024/08/07        */
/***************************/
'''
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from solar_system_body import Body
import sys
from scipy.integrate import solve_ivp

from types import SimpleNamespace

cfg = SimpleNamespace(
    # time_span=(0, 3.154e7),  # 1 year in seconds
    time_span=(0, 6.308e7),    # 2 year in seconds
    useRK45=True,              # use RK45
    step=86400,                # integration step 1 day
    verbose=True,              # verbose output
    yukawa_coeff=2e15          # Yukawa potential coefficient
)


cst = SimpleNamespace(
    G=6.67430e-11
)

c = SimpleNamespace(
    b=(102 / 255, 204 / 255, 255 / 255),  # blue
    o=(255 / 255, 153 / 255, 102 / 255),  # orange
    r=(204 / 255, 0 / 255, 102 / 255),    # red
    g=(102 / 255, 204 / 255, 102 / 255)   # green
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


def n_body_problem(t, y, masses, gravitational_force):
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
    def __init__(self, odir):
        self._bodies = []
        self._num_frames = 30
        self._odir = odir
        self._use_newton = True


    @property
    def useNewtown(self):
        return self._use_newton

    @useNewtown.setter
    def useNewtown(self, useNewtown : bool):
        self._use_newton = useNewtown

    @property
    def bodies(self):
        return self._bodies

    def create_bodies(self):
        with open('data_input/bodies_yukawa.json', 'r') as file:
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
        gravitational_force = gravitational_force_newton if self._use_newton \
            else gravitational_force_yukawa
        if cfg.useRK45:
            # max_step is needed otherwise Rk45 is not coverging
            solution = solve_ivp(
                n_body_problem, cfg.time_span, initial_conditions,
                args=(masses, gravitational_force), method='RK45',
                t_eval=t_eval, max_step=cfg.step)
            self._positions = solution.y[:num_bodies * 3].reshape(
                (num_bodies, 3, -1))
            self._velocities = solution.y[num_bodies * 3:].reshape(
                (num_bodies, 3, -1))
        else:
            t, y = runge_kutta4(n_body_problem, initial_conditions,
                                cfg.time_span[0], cfg.time_span[1],
                                cfg.step, args=(masses, gravitational_force))
            self._positions = y[:num_bodies * 3].reshape((num_bodies, 3, -1))
            self._velocities = y[num_bodies * 3:].reshape((num_bodies, 3, -1))

    def compute_distances(self):
        # calculate distances from the Sun
        r = np.sqrt(self._positions[1, 0, :]**2 +
                    self._positions[1, 1, :]**2 +
                    self._positions[1, 2, :]**2)
        # calculate velocities
        # find perihelion (minimum distance) and aphelion (maximum distance)
        r_p = np.min(r)
        r_a = np.max(r)
        print(f"Perihelion: {r_p:.8e}")
        print(f"Aphelion: {r_a:.8e}")

    def create_axes(self, fig):
        ax = fig.add_subplot(111)
        ax.axis("on")
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.grid(linewidth=0.4, linestyle="--", dashes=(5, 20))
        return ax

    def plot_orbit(self):
        # calculate the normal vector to the plane
        normal = np.cross(self._bodies[1].position, self._bodies[1].velocity)
        # normalize the normal vector
        normal /= np.linalg.norm(normal)

        # define two vectors in the plane (orthonormal basis)
        if normal[0] == 0 and normal[1] == 0:
            plane_vector1 = np.array([1, 0, 0])
        else:
            plane_vector1 = np.cross(normal, [0, 0, 1])
        plane_vector1 /= np.linalg.norm(plane_vector1)
        plane_vector2 = np.cross(normal, plane_vector1)
        plane_vector2 /= np.linalg.norm(plane_vector2)

        # project the points onto the plane
        positions_3d = np.vstack(
            (self._positions[1, 0, :], self._positions[1, 1, :],
             self._positions[1, 2, :])).T
        projection_matrix = np.vstack((plane_vector1, plane_vector2)).T
        projected_points = positions_3d @ projection_matrix

        # extract the projected coordinates
        x_projected = projected_points[:, 0]
        y_projected = projected_points[:, 1]

        # plot the projected 2D orbits
        fig = plt.figure()
        ax = self.create_axes(fig)
        ax.set_aspect('equal', adjustable='box')
        ax.plot(x_projected, y_projected, color=c.b, linewidth=2.0)
        ax.scatter([0], [0],color=c.o, s=400, label='Sun', marker='o',
                   edgecolors='k')
        ax.scatter(x_projected[0], y_projected[0],color=c.g, s=150,
                label='Earth', marker='o', edgecolors='k', zorder=2)
        plt.tight_layout()
        gname = 'newtown' if self._use_newton else 'yukawa'
        plt.savefig(self._odir + f"/earth_orbit_{gname}.png", dpi=300)
        plt.show()

    def plot_energy(self):
        # calculate distances from the Sun
        r = np.sqrt(self._positions[1, 0, :]**2 +
                    self._positions[1, 1, :]**2 +
                    self._positions[1, 2, :]**2)
        # calculate velocities
        v = np.sqrt(self._velocities[1, 0, :]**2 +
                    self._velocities[1, 1, :]**2 +
                    self._velocities[1, 2, :]**2)

        # calculate kinetic energy
        KE = 0.5 * self.bodies[1].mass * v**2
        # calculate potential energy
        PE = -cst.G * self.bodies[0].mass * self.bodies[1].mass / r
        # calculate total energy
        TE = KE + PE

        # Plot the energies
        fig = plt.figure()
        ax = self.create_axes(fig)
        ax.plot(KE, label='Kinetic Energy', color=c.b, linewidth=2.0)
        ax.plot(PE, label='Potential Energy', color=c.o, linewidth=2.0)
        ax.plot(TE, label='Total Energy', color=c.r, linewidth=3.0)
        ax.set_ylabel('Energy ($J$)', fontsize=15)
        ax.legend()
        ax.set_xticklabels([])
        plt.tight_layout()
        plt.savefig(self._odir + '/energies.png', dpi=300)
        plt.show()

        # compute the theoretical total energy
        # calculate the semi-major axis
        r_p = np.min(r)
        r_a = np.max(r)
        a = (r_a + r_p) / 2

        # Calculate the total energy
        E_T = -cst.G * self.bodies[0].mass * self.bodies[1].mass / (2 * a)
        print(f"Total Energy 2 bodies: {E_T:.3e} Joules")
        print(f"Calculated Energy 2 bodies: {TE[0]:.3e} Joules")

        # print(f"Total Energy: {E} Joules")
        DeltaTE = (TE - TE[0]) / E_T
        # plot the relative change in total energy
        fig = plt.figure()
        ax = self.create_axes(fig)
        ax.plot(DeltaTE, label='Relative Change in Total Energy', color=c.r,
                linewidth=2.0)
        ax.set_ylabel('$\\%$ Change in Total Energy', fontsize=15)
        ax.set_xticklabels([])
        plt.tight_layout()
        plt.savefig(self._odir + '/delta_E.png', dpi=300)
        plt.show()


def make_plot(odir: str):
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} " \
        r"\usepackage{amsmath} \usepackage{helvet}"
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica"
    })
    plt.rcParams['animation.convert_path'] = 'magick'

    sim = SolarSystemSimulation(odir)
    sim.create_bodies()
    sim.compute()
    sim.compute_distances()
    sim.plot_orbit()
    sim.plot_energy()
    sim.useNewtown = False
    sim.compute()
    sim.compute_distances()
    sim.plot_orbit()


def main():
    parser = argparse.ArgumentParser(
        description='solar system simulation')
    parser.add_argument('-o', '--odir', help='output dir')
    args = parser.parse_args()
    if args.odir:
        odir = args.odir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        odir = tmp_dir
    make_plot(odir)


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    main()
