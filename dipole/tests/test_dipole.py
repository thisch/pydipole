from __future__ import print_function

import logging
import numpy as np
import pytest
import matplotlib.pyplot as plt

from ..field import dipole_e_ff
from ..field import dipole_general
from ..field import dipole_radiant_intensity
from ..helper import gen_r
from .base import Base

LG = logging.getLogger('dip')


def main(test, ndip, k=0.08, ngrid=100, thetamax=45.,
         rdisk=None, aligned_dipoles=False, align_axis='z', onsphere=True,
         s=None, plot=True):
    Lam = 2*np.pi/k
    reval = 1000*Lam
    reval2 = 5.*Lam
    LG.info('#### SETTINGS: #dips=%d, k=%g, reval=%g', ndip, k, reval)

    rparams = gen_r(ngrid, reval=reval, onsphere=onsphere, thetamax=thetamax)
    if rdisk:
        rparams2 = gen_r(ngrid, reval=reval2, onsphere=False,
                         rmax=rdisk*1.2)

    if s is not None:
        rand = s.rand
    else:
        rand = np.random.rand

    pdisk = 2.*(rand(ndip, 3) - .5)  # dipole moments
    if aligned_dipoles:
        if align_axis == 'xy':
            pdisk[:, 2] = 0.
        else:
            pdisk = np.zeros((ndip, 3))
            if align_axis == 'z':
                pdisk[:, 2] = 1.
            elif align_axis == 'x':
                pdisk[:, 0] = 1.
            elif align_axis == 'y':
                pdisk[:, 1] = 1.

    if ndip == 1:
        rdip = np.zeros((ndip, 3))
        phases = np.zeros(ndip)
    else:
        rdip = (rand(ndip, 3) - .5)*2*rdisk  # dipol aufpunkte
        # LG.info("rdip/rdisk %s", rdip[0,:]/rdisk)
        rdip[:, 2] = 0.  # all dipoles lay in the xy-plane
        phases = rand(ndip) * 2*np.pi

        # ax.plot(rdip[:, 0], rdip[:, 1], 'x', label='k=%g' % kcur)
    if onsphere:
        intens = dipole_radiant_intensity(rparams[0],
                                          rparams[1],
                                          pdisk, rdip, phases, k)
    else:
        intens = np.linalg.norm(
            dipole_e_ff(rparams[-1], pdisk, rdip, phases, k, t=0),
            axis=2)**2

    if rdisk is not None:
        tot2, _ = dipole_general(rparams2[-1], pdisk, rdip,
                                 phases, k, t=0)
        return reval, rparams + (tot,), reval2, rparams2 + (tot2,)
    return reval, rparams + (tot,)


class TestAnalytic(Base):

    @pytest.mark.parametrize(('k', 'onsphere'),
                             ([0.01, True],
                              [0.01, False],
                              [1., True],
                              [1., False]))
    def test_single(self, k, onsphere):
        """
        single dipole
        """
        if not onsphere:
            thetamax = 40.
        else:
            thetamax = 90.
        for align in 'xyz':
            reval, iparams = main(self, ndip=1, aligned_dipoles=True,
                                  thetamax=thetamax, align_axis=align,
                                  onsphere=onsphere, k=k, ngrid=200)
            if onsphere:
                T, P, _, intens = iparams
                self._plot_intens(T, P, intens=intens)
            else:
                X, Y, _, intens = iparams
                self._plot_intens(intens=intens, XY=(X, Y))
            ax = plt.gca()
            ax.set_title('k=%g, dipole orientation: %s-axis' % (k, align))
        self.show()

    def _polarization_main(self, ndip, onsphere=False, k=1., rdisk=None, s=None,
                           alignments=None):
        if not onsphere:
            thetamax = 14
        else:
            thetamax = 20.
        if alignments is None:
            alignments = 'xyz'
        for align in alignments:
            ret = main(self, ndip=ndip, aligned_dipoles=True,
                       rdisk=rdisk, thetamax=thetamax, align_axis=align, s=s,
                       onsphere=onsphere, k=k, ngrid=32)
            if rdisk is None:
                reval, fparams = ret
            else:
                reval, fparams, reval2, fparams2 = ret

            if onsphere:
                T, P, _, field = fparams
                e_t = np.zeros((3,) + P.shape)
                e_t[0, :, :] = np.cos(T)*np.cos(P)
                e_t[1, :, :] = np.cos(T)*np.sin(P)
                e_t[2, :, :] = -np.sin(T)

                # phi unit vec
                e_p = np.zeros((3,) + P.shape)
                e_p[0, :, :] = -np.sin(P)
                e_p[1, :, :] = np.cos(P)

                # TODO project Ex, Ey onto etheta, ephi?
                Ex = field[:, :, 0].real
                Ey = field[:, :, 1].real
                fig, ax = plt.subplots()
                ax.quiver(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
                          Ex.real, Ey.real)
                ax.set_xlabel('tx [deg]')
                ax.set_ylabel('ty [deg]')
            else:
                X, Y, _, field = fparams
                tx = np.degrees(np.arctan2(X, reval))
                ty = np.degrees(np.arctan2(Y, reval))

                Ex = field[:, :, 0]
                Ey = field[:, :, 1]
                fig, ax = plt.subplots()
                ax.quiver(tx, ty, Ex.real, Ey.real)
                ax.set_xlabel('tx')
                ax.set_ylabel('ty')
                ax.set_title('Ex, Ey:  k=%g, orientation of dip. moment : '
                             '%s-axis' % (k, align))

            polangles = np.linspace(0, 180, 64)
            phis = np.radians(polangles)
            Is = np.array([(abs(np.cos(phi)*Ex +
                                np.sin(phi)*Ey)**2).sum() for phi in phis])
            Is /= Is.max()
            fig, ax = plt.subplots()
            ax.plot(polangles, Is, '-', label='pol farfield')

            if rdisk is not None:
                field = fparams2[-1]
                Ex, Ey = field[:, :, 0], field[:, :, 1]
                Is = np.array([(abs(np.cos(phi)*Ex +
                                    np.sin(phi)*Ey)**2).sum() for phi in phis])
                Is /= Is.max()
                ax.plot(polangles, Is, '-', label='pol nearfield')
            ax.legend(fontsize=8)

    def test_single_dp_pol(self):
        """
        polarization in the far-field of a single oscillating dipole
        """
        self._polarization_main(ndip=1, k=0.08)
        self.show()

    def test_many_dp_pol(self):
        """
        polarization in the far-field of many oscillating dipoles
        """
        s = np.random.RandomState(12321)
        self._polarization_main(ndip=150, k=0.08, rdisk=250,
                                s=s, alignments='x')
                                # alignments=['xy'])
        self.show()

    # @pytest.mark.xfail
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

        ts = np.linspace(0., Tper*2, 60)
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

        # def _blit_draw(self, artists, bg_cache):
        #     # Handles blitted drawing, which renders only the artists given instead
        #     # of the entire figure.
        #     updated_ax = []
        #     for a in artists:
        #         # If we haven't cached the background for this axes object, do
        #         # so now. This might not always be reliable, but it's an attempt
        #         # to automate the process.
        #         if a.axes not in bg_cache:
        #             # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
        #             # change here
        #             bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        #         a.axes.draw_artist(a)
        #         updated_ax.append(a.axes)

        #     # After rendering all the needed artists, blit each axes individually.
        #     for ax in set(updated_ax):
        #         # and here
        #         # ax.figure.canvas.blit(ax.bbox)
        #         ax.figure.canvas.blit(ax.figure.bbox)

        from matplotlib import animation
        # MONKEY PATCH!!
        # animation.Animation._blit_draw = _blit_draw

        levels = np.linspace(0, 8e17, 25)
        # levels = np.linspace(0, 8e20, 25)
        # levels = None

        def animate(i):
            self.log.info('[%d/%d] ', i+1, len(ts))
            z = 0.5*np.cross(Eres,
                             Hres.conjugate() + Hres*np.exp(-2j*k*co.c*ts[i])).real
            # res = dipole_e_ff(r, pring, rring, phases, k=21, t=ts[i])
            # z = (Eres*np.exp(-1j*k*co.c*ts[i])).real
            z = np.linalg.norm(z, axis=2)
            print('z.max(): %g' % z.max())
            # qvs = ax.contourf(np.degrees(self.tx), np.degrees(self.ty), z, 25)
            qvs = ax.contourf(X, Y, z, levels=levels)
            ax.set_title('t = %g' % (ts[i]/Tper))
            return qvs

        ims = []
        for t in ts:
            z = 0.5*np.cross(Eres,
                             Hres.conjugate() + Hres*np.exp(-2j*k*co.c*t)).real
            z = np.linalg.norm(z, axis=2)
            # print('z.max(): %g' % z.max())
            qvs = ax.contourf(X, Y, z, levels=levels, extend='both')
            qvs.cmap.set_over(qvs.cmap.colors[-1])
            # FIX QuadContourSet does not implement the Artist interface
            import types

            def setvisible(self, vis):
                for c in self.collections:
                    c.set_visible(vis)

            qvs.set_visible = types.MethodType(setvisible, qvs)

            def setanimated(self, an):
                for c in self.collections:
                    c.set_animated(an)

            qvs.set_animated = types.MethodType(setanimated, qvs)
            qvs.axes = ax
            qvs.figure = fig
            qvs.draw = qvs.axes.draw

            # FIXME title does not get updated
            # tit = ax.set_title('t = %g' % (t/Tper))
            tit = ax.text(0.5, 1.04, r'$\nu t = %g$' % (t/Tper),
                          transform=ax.transAxes, ha='center', va='top',
                          fontsize=14)

            ims.append((qvs, tit))

        # animate(0)
        # self.show()

        # ani = animation.FuncAnimation(fig, animate, frames=len(ts))
        ani = animation.ArtistAnimation(fig, ims, blit=False, interval=250)
        # ani.save('single.gif')  # see issue 5592
        self.show()

    @pytest.mark.parametrize('parallel', [True, False])
    def test_2parallel(self, parallel):
        """ 2 parallel dipoles
        """
        ngrid = 256
        k = 1.
        La = 2*np.pi/k
        ndip = 2

        T, P, r = gen_r(ngrid, onsphere=True, reval=500., thetamax=90.)

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

    def test_2antiparallel(self):
        ngrid = 256
        k = 1.
        La = 2*np.pi/k
        ndip = 2
        reval = 1000*La
        T, P, r = gen_r(ngrid, onsphere=True, reval=reval, thetamax=90.)

        pdisk = np.zeros((ndip, 3))
        pdisk[:, 2] = 1.
        pdisk[1, 2] = -1.

        for distfac in [0.1, 1, 10, 100, 1000]:
            rdip = np.zeros((2, 3))
            rdip[1, 0] = distfac*La/2
            rdip[0, 0] = -distfac*La/2

            phases = np.array([0., 0.])
            res = dipole_e_ff(r, pdisk, rdip, phases, k, 0)
            self._plot_intens(T, P, res, title='distfac=%g' % distfac)
        self.show()
