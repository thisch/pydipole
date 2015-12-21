from __future__ import print_function

import cmath
import logging
import numpy as np
import pytest

from ..field import dipole_e_ff
from ..field import dipole_general
from ..helper import unit_vectors
from .base import Base

LG = logging.getLogger('dip')

rextrafac = 100.


def main(test, diameter, ndip, k=0.08, ngrid=100, thetaopen=45.,
         aligned_dipoles=False, align_axis='z', onsphere=True,
         plot=True, seed=12345):
    LG.info('#### diameter disk=%g, #dips=%d, k=%s', diameter, ndip, k)
    LG.debug('spherical grid points: %d' % ngrid)

    np.random.seed(seed)
    rdisk = diameter/2.
    z0 = rdisk*rextrafac  # z coord of xy plane
    rmax = np.tan(np.radians(thetaopen)) * z0

    P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=thetaopen,
                                         ngrid=ngrid)
    r = np.empty((ngrid, ngrid, 3))
    if onsphere:
        r[:, :, 0] = z0 * e_r[0, :, :]
        r[:, :, 1] = z0 * e_r[1, :, :]
        r[:, :, 2] = z0 * e_r[2, :, :]
    else:
        LG.info(" %s deg", np.degrees(cmath.phase(z0 + 1j*np.sqrt(2)*rmax)))
        rng = np.linspace(-rmax, rmax, ngrid)
        X, Y = np.meshgrid(rng, rng)
        r[:, :, 0] = X
        r[:, :, 1] = Y
        r[:, :, 2] = z0
    LG.debug("onsphere: %s\tfirst r vec %s", onsphere, r[0, 0, :])

    tot = np.zeros(r.shape, dtype='complex128')
    res = np.empty(r.shape, dtype='complex128')

    try:
        k[0]
    except TypeError:
        k = [k]

    for kcur in k:
        LG.info('kcur %g', kcur)
        dipole_phis = np.random.rand(ndip) * 2*np.pi
        pdisk = 2.*(np.random.rand(ndip, 3) - .5)  # dipole moments
        if aligned_dipoles:
            pdisk = np.zeros((ndip, 3))
            if align_axis == 'z':
                pdisk[:, 2] = 1.
            elif align_axis == 'x':
                pdisk[:, 0] = 1.
            elif align_axis == 'y':
                pdisk[:, 1] = 1.

        rdip = (np.random.rand(ndip, 3) - .5)*2*rdisk  # dipol aufpunkte
        # LG.info("rdip/rdisk %s", rdip[0,:]/rdisk)
        rdip[:, 2] = 0.  # all dipoles lay in the xy-plane
        if ndip == 1:
            rdip[:, :] = 0.  # dipole at the origin
        # ax.plot(rdip[:, 0], rdip[:, 1], 'x', label='k=%g' % kcur)

        phases = np.r_[dipole_phis]
        t0 = 0
        res = dipole_e_ff(r, pdisk, rdip, phases, kcur, t0)
        tot += res
    return T, P, tot


class TestAnalytic(Base):

    @pytest.mark.parametrize('k', [0.01, 1.0])
    def test_single(self, k):
        """
        single dipole
        """

        k = 0.01

        for align in 'xyz':
            T, P, field = main(self, diameter=100, ndip=1, aligned_dipoles=True,
                               thetaopen=90, align_axis=align,
                               k=[k], ngrid=200)
            self._plot_intens(T, P, field)
        self.show()

    def test_single_anim(self):
        """
        animation of an oscillating dipole in the xy-plane
        """
        k = 1.  # todo copy and pasted
        import scipy.constants.constants as co
        omega = k*co.c
        Tper = 2*np.pi/omega  # max. time

        ngrid = 256
        La = 2*np.pi/k
        ndip = 1

        r = np.empty((ngrid, ngrid, 3))
        # thetamax = 15.
        # z0 = 100.*La
        # self.log.info('zeval/lambda: %g', z0/La)
        # P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=thetamax, ngrid=ngrid)
        # self.tx = T*np.cos(P)
        # self.ty = T*np.sin(P)
        # for i in range(3):
        #     r[:, :, i] = z0 * e_r[i, :, :]
        rmax = 1*La
        rng = np.linspace(-rmax, rmax, ngrid)
        X, Y = np.meshgrid(rng, rng)
        r[:, :, 0] = X
        r[:, :, 1] = Y
        r[:, :, 2] = 0  # 0.05*La

        pring = np.zeros((ndip, 3))  # dipole moments
        pring[0, 1] = 1.  # y-Axis

        rring = np.zeros((ndip, 3))  # dipol aufpunkte
        phases = np.zeros(ndip)
        from dipole.utils import Timer
        with Timer(self.log.debug, ('dipole_general() (%d points) took '
                                    '%%f ms') % (r.shape[0]*r.shape[1])):
            Eres, Hres = dipole_general(r, pring, rring, phases, k, t=0)

        ts = np.linspace(0., Tper*2, 30)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # ax.set_xlim(-thetamax, thetamax)
        # ax.set_ylim(-thetamax, thetamax)
        # ax.set_xlabel('thetax [deg]')
        # ax.set_ylabel('thetay [deg]')
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        levels = np.linspace(0, 8e17, 25)
        # levels = np.linspace(0, 8e20, 25)
        # levels = None

        def animate(i):
            self.log.info('[%d/%d] ', i+1, len(ts))
            z = 0.5*np.cross(Eres,
                             Hres.conjugate() + Hres*np.exp(2j*k*co.c*ts[i])).real
            # res = dipole_e_ff(r, pring, rring, phases, k=21, t=ts[i])
            # z = (Eres*np.exp(-1j*k*co.c*ts[i])).real
            z = np.linalg.norm(z, axis=2)
            print('z.max(): %g' % z.max())
            # qvs = ax.contourf(np.degrees(self.tx), np.degrees(self.ty), z, 25)
            qvs = ax.contourf(X, Y, z, levels=levels)
            ax.set_title('t = %g' % (ts[i]/Tper))
            return qvs

        # animate(0)
        # self.show()

        from matplotlib import animation
        ani = animation.FuncAnimation(fig, animate, frames=len(ts))
        self.show()

    @pytest.mark.parametrize('parallel', [True, False])
    def test_2parallel(self, parallel):
        """ 2 parallel dipoles
        """
        ngrid = 256
        k = 1.
        La = 2*np.pi/k
        ndip = 2
        P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=90.,
                                             ngrid=ngrid)
        r = np.empty((ngrid, ngrid, 3))
        z0 = 500.
        LG.info("reval/lambda = %g", z0/La)
        r[:, :, 0] = z0 * e_r[0, :, :]
        r[:, :, 1] = z0 * e_r[1, :, :]
        r[:, :, 2] = z0 * e_r[2, :, :]
        LG.debug("first r vec %s", r[0, 0, :])

        res = np.empty(r.shape, dtype='complex128')

        pdisk = np.zeros((ndip, 3))
        pdisk[:, 2] = 1.
        if not parallel:
            pdisk[1, 2] = -1.

        rdip = np.zeros((2, 3))
        rdip[1, 0] = La/2
        rdip[0, 0] = -La/2

        phases = np.array([0., 0.])
        res = dipole_e_ff(r, pdisk, rdip, phases, k, 0)
        resE, resH = dipole_general(r, pdisk, rdip, phases, k, 0)
        Smean = 0.5*np.cross(resE, resH).real

        self._plot_intens(T, P, res)
        self._plot_poynting(T, P, S=np.linalg.norm(Smean, axis=2),
                            title='poynting')
        self.show()
