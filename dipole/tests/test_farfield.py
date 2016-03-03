"""
use the pydipole functions dipole_radiant_intensity, dipole_field_general and
dipole_e_ff and check that the radiant intensities are equal when computed with
these functions.
"""
from __future__ import print_function

import logging
import numpy as np
import numpy.testing as nt
import matplotlib.pyplot as plt

from ..field import dipole_e_ff
from ..field import dipole_general
from ..field import dipole_radiant_intensity
from ..helper import gen_r
from ..utils import Timer
from .base import Base

LG = logging.getLogger('dip')
# TODO move to dipole.constants
c = 299792458.
mu0 = 4*np.pi*1e-7
Z = mu0*c


def plot(expr, T, P, title):
    fig, ax = plt.subplots()
    C = ax.pcolormesh(np.degrees(T*np.cos(P)),
                      np.degrees(T*np.sin(P)), expr)
    ax.set_aspect('equal')
    plt.colorbar(C)
    ax.set_title(title)


class TestFarField(Base):
    def test_1(self):
        k = 0.08
        Lam = 2*np.pi/k
        reval = 1e6*Lam
        ngrid = 256
        T, P, r = gen_r(ngrid, onsphere=True, reval=reval, thetamax=90.)
        ndip = 1
        pdisk = np.zeros((ndip, 3))
        pdisk[0, 2] = -1
        rdip = np.zeros((ndip, 3))
        phases = np.array([0.])

        with Timer(self.log.debug, 'dp radint took: %f ms'):
            rint = dipole_radiant_intensity(T, P, pdisk, rdip, phases, k)
        plot(rint, T, P, 'rint = r^2*smean [dipole radiant int]')

        with Timer(self.log.debug, 'dp general took: %f ms'):
            resE, resH = dipole_general(r, pdisk, rdip, phases, k, 0)
        Smean = 0.5*np.cross(resE, resH.conjugate()).real
        Smean = np.linalg.norm(Smean, axis=2)
        rint1 = reval**2 * Smean
        plot(rint1, T, P, 'r^2*smean [dipole general]')
        # print(abs(rint).max())
        # print(abs(rint - rint1))
        # print(abs(rint - rint1).max())
        nt.assert_allclose(rint, rint1, atol=2e4, rtol=0)

        with Timer(self.log.debug, 'dp e ff took: %f ms'):
            res = dipole_e_ff(r, pdisk, rdip, phases, k, 0)
        esqrd = np.linalg.norm(res, axis=2)**2  # |E|^2
        rint2 = reval**2/(2*Z)*esqrd
        plot(rint2, T, P, 'r^2/(2Z)*|E|^2 [dipole e ff]')

        # print(abs(rint - rint2))
        # print(abs(rint - rint2).max())
        nt.assert_allclose(rint, rint2, atol=1e4, rtol=0)

        self.show()

    def test_1_varyreval(self):
        k = 0.08
        Lam = 2*np.pi/k
        ngrid = 256
        ndip = 1
        pdisk = np.zeros((ndip, 3))
        pdisk[0, 2] = -1
        rdip = np.zeros((ndip, 3))
        phases = np.array([0.])

        # TODO test dipole_e_ff and dipole_general
        reval = 1e6*Lam
        T, P, r = gen_r(ngrid, onsphere=True, reval=reval, thetamax=90.)
        with Timer(self.log.debug, 'dp radint took: %f ms'):
            rint6 = dipole_radiant_intensity(T, P, pdisk, rdip, phases, k)
        plot(rint6, T, P, 'rint = r^2*smean [revfac 1e6]')

        reval = 1e7*Lam
        T, P, r = gen_r(ngrid, onsphere=True, reval=reval, thetamax=90.)
        with Timer(self.log.debug, 'dp radint took: %f ms'):
            rint7 = dipole_radiant_intensity(T, P, pdisk, rdip, phases, k)
        plot(rint7, T, P, 'rint = r^2*smean [revfac 1e7]')

        nt.assert_allclose(rint6, rint7, atol=2e4, rtol=0)

        self.show()

    def test_2(self):
        """ differnt dipole configuration than in 1
        """
        k = 0.08
        Lam = 2*np.pi/k
        reval = 1e6*Lam
        ngrid = 256
        T, P, r = gen_r(ngrid, onsphere=True, reval=reval, thetamax=90.)

        ndip = 2
        pdisk = np.zeros((ndip, 3))
        pdisk[0, 2] = 1.
        pdisk[1, 2] = -1.

        rdip = np.zeros((ndip, 3))
        rdip[0, 0] = -Lam/2
        rdip[1, 0] = Lam/2

        phases = np.array([0., 0.])

        rint = dipole_radiant_intensity(T, P, pdisk, rdip, phases, k)
        plot(rint, T, P, 'rint = r^2*smean [dipole radiant int]')

        resE, resH = dipole_general(r, pdisk, rdip, phases, k, 0)
        Smean = 0.5*np.cross(resE, resH.conjugate()).real
        Smean = np.linalg.norm(Smean, axis=2)
        rint1 = reval**2 * Smean
        plot(rint1, T, P, 'r^2*smean [dipole general]')
        # print(abs(rint).max())
        # print(abs(rint - rint1))
        # print(abs(rint - rint1).max())
        nt.assert_allclose(rint, rint1, atol=2e4, rtol=0)

        res = dipole_e_ff(r, pdisk, rdip, phases, k, 0)
        esqrd = np.linalg.norm(res, axis=2)**2  # |E|^2
        rint2 = reval**2/(2*Z)*esqrd
        plot(rint2, T, P, 'r^2/(2Z)*|E|^2 [dipole e ff]')

        # print(abs(rint - rint2))
        # print(abs(rint - rint2).max())
        nt.assert_allclose(rint, rint2, atol=1e4, rtol=0)

        self.show()

    def test_ffevalradius(self):
        pass
