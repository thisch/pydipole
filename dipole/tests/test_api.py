from __future__ import print_function

import logging
from timeit import default_timer

import numpy as np
import numpy.testing as nt
import matplotlib.pyplot as plt

from ..helper import gen_r
from .base import Base

LG = logging.getLogger('dip')


class Timer:
    def __init__(self, stream, msg=None):
        super().__init__()
        self.msg = msg
        self.stream = stream
        self.timer = default_timer

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        if self.msg:
            self.stream(self.msg % self.elapsed)
        else:
            self.stream('elapsed time: %f ms' % self.elapsed)


class TestAPI(Base):
    def test_rint(self):
        k, rdisk = 0.08, 250.
        ndip = 6
        thetamax = 60
        ngrid = 200
        # s = None
        s = np.random.RandomState(12321)
        # use dipole_general to calc E,H in the ff
        # compare with dipole_radiant_int
        # dipole_e_ff, dipole_h_ff

        Lam = 2*np.pi/k
        reval = 1e6*Lam
        LG.info('#### SETTINGS: #dips=%d, k=%g, reval=%g', ndip, k, reval)

        rparams = gen_r(ngrid, reval=reval, onsphere=True, thetamax=thetamax)
        if s is not None:
            rand = s.rand
        else:
            rand = np.random.rand

        pdisk = 2.*(rand(ndip, 3) - .5)  # dipole moments
        if ndip == 1:
            rdip = np.zeros((ndip, 3))
            phases = np.zeros(ndip)
        else:
            rdip = (rand(ndip, 3) - .5)*2*rdisk  # dipol aufpunkte
            # LG.info("rdip/rdisk %s", rdip[0,:]/rdisk)
            rdip[:, 2] = 0.  # all dipoles lay in the xy-plane
            phases = rand(ndip) * 2*np.pi

        T, P = rparams[0], rparams[1]

        # radial unit vec
        e_r = np.zeros(P.shape + (3,))
        e_r[:, :, 0] = np.sin(T)*np.cos(P)
        e_r[:, :, 1] = np.sin(T)*np.sin(P)
        e_r[:, :, 2] = np.cos(T)

        from dipole.field import (dipole_radiant_intensity,
                                  dipole_e_ff,
                                  dipole_h_ff,
                                  dipole_general)
        ## 1
        with Timer(LG.info, 'dri took %f ms'):
            intens_ri = dipole_radiant_intensity(T, P, pdisk,
                                                 rdip, phases, k)

        ## 2
        field_e = dipole_e_ff(rparams[-1], pdisk, rdip, phases, k, t=0)
        field_h = dipole_h_ff(rparams[-1], pdisk, rdip, phases, k, t=0)
        s_ff_mean = 0.5*np.cross(field_e, field_h.conjugate()).real
        # dot product with e_r
        intens_ri_ff = np.sum(e_r*s_ff_mean, axis=2) * reval**2
        # print('e', field_e[0,0,:])
        # print(field_h[0,0,:])
        # print(np.cross(field_e[0,0,:],
        #                field_h[0,0,:].conjugate()))

        ## 3
        fg_e, fg_h = dipole_general(rparams[-1], pdisk, rdip, phases, k, t=0)
        s_g_mean = 0.5*np.cross(fg_e, fg_h.conjugate()).real
        # dot product with e_r
        intens_ri_g = np.sum(e_r*s_g_mean, axis=2) * reval**2
        # print('e',  fg_e[0,0,:])
        # print(fg_h[0,0,:])
        # print(np.cross(fg_e[0,0,:],
        #                fg_h[0,0,:].conjugate()))

        nt.assert_allclose(intens_ri, intens_ri_g, atol=8e8, rtol=0)
        nt.assert_allclose(intens_ri, intens_ri_ff, atol=8e8, rtol=0)

        nt.assert_allclose(field_e, fg_e, atol=1e-4, rtol=0)
        nt.assert_allclose(field_h, fg_h, atol=1e-4, rtol=0)
        # nt.assert_allclose(field_e, fg_e, atol=0., rtol=1e-1)
        # nt.assert_allclose(field_h, fg_h, atol=0., rtol=1e-1)

        tx, ty = T*np.cos(P), T*np.sin(P)
        plt.subplots()
        cax = plt.contourf(np.degrees(tx), np.degrees(ty), intens_ri)
        plt.colorbar(cax)
        plt.title('intensri')
        plt.subplots()
        cax = plt.contourf(np.degrees(tx), np.degrees(ty), intens_ri_ff)
        plt.colorbar(cax)
        plt.title('intensri ff')
        plt.subplots()
        cax = plt.contourf(np.degrees(tx), np.degrees(ty), intens_ri_g)
        plt.colorbar(cax)
        plt.title('intensri g')

        # err_ri = np.abs(intens_ri - intens_ri_g)
        # plt.subplots()
        # cax = plt.contourf(np.degrees(tx), np.degrees(ty), err_ri)
        # plt.colorbar(cax)
        # plt.title('err')

        self.show()
