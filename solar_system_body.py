#!/usr/bin/env python3
'''
/************************/
/* solar_system_body.py */
/*    Version 1.0       */
/*     2024/08/03       */
/************************/
'''
import numpy as np
import sys


class Body:
    def __init__(self, name='', radius=0, mass=0, scale=1,
                 color=np.zeros(3),
                 position=np.zeros(3), velocity=np.zeros(3)):
        self._name = name
        self._radius = radius
        self._mass = mass
        self._scale = scale
        self._color = color
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
    def mass_plot(self):
        return np.log(self._mass) / np.log(self._scale)

    @property
    def color(self):
        return self._color


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


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    pass
