import pytest
import numpy as np

from ..helper import unit_vectors
from ..field import dipole_e_ff
from ..field import dipole_general
from ..utils import Timer

from .base import Base
from .mixins import FFTMixin

import matplotlib.pyplot as plt


class TestRing(Base, FFTMixin):
    def _rolf(self, sphere, shift, spp, rringfac=15, thetamax=70,
              horizontal_line=False, t=0, show=True, ax=None, ngrid=256,
              spp_ramp_height=2*np.pi):

        # FIXME implement amplitude correction if sphere=False and apply it
        # if desired
        k = 1.
        La = 2*np.pi/k
        ndip = 256
        z0 = 500.*La
        r = np.empty((ngrid, ngrid, 3))
        self.log.info('zeval/lambda: %g', z0/La)
        if horizontal_line:
            # on a sphere
            r = np.empty((1, ngrid, 3))
            tlsp = np.radians(np.linspace(-thetamax, thetamax, ngrid))
            r[0, :, 0] = z0 * np.sin(tlsp)
            r[0, :, 1] = 0
            r[0, :, 2] = z0 * np.cos(tlsp)
        elif sphere:
            P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=thetamax,
                                                 ngrid=ngrid)
            for i in range(3):
                r[:, :, i] = z0 * e_r[i, :, :]
        else:
            thetamax = min(15., thetamax)
            tm = np.radians(thetamax)
            rmax = z0 * np.tan(tm)
            rng = np.linspace(-rmax, rmax, ngrid)
            X, Y = np.meshgrid(rng, rng)
            r[:, :, 0] = X
            r[:, :, 1] = Y
            r[:, :, 2] = z0

        res = np.empty(r.shape, dtype='complex128')

        dipole_phis, dphi = np.linspace(0, 2*np.pi, ndip,
                                        retstep=True, endpoint=False)
        ind = (dipole_phis > np.pi/2) & (dipole_phis < 3*np.pi/2)
        # dipole_phis[ind] += dphi/2.

        pringfac = 1.
        pring = np.zeros((ndip, 3))  # dipole moments
        ringslice = slice(None, None, None)
        pring[ringslice, 0] = -np.sin(dipole_phis) * pringfac
        pring[ringslice, 1] = np.cos(dipole_phis) * pringfac

        rfac = rringfac*La
        rring = np.zeros((ndip, 3))  # dipol aufpunkte
        rring[ringslice, 0] = np.cos(dipole_phis) * rfac
        rring[ringslice, 1] = np.sin(dipole_phis) * rfac

        # fig, ax = plt.subplots()
        # ax.plot(rring[:, 0],
        #         rring[:, 1], 'x')
        # ax.grid(True)

        phases = np.zeros(ndip)
        if shift:
            phases[ind] = np.pi
        if spp:
            l = 1
            phases = np.linspace(0, l*spp_ramp_height, ndip)

        general = True
        static = False
        if static:
            with Timer(self.log.debug, ('dipole_general() (%d points) took '
                       '%%f ms') % (r.shape[0]*r.shape[1])):
                Eres, Hres = dipole_general(r, pring, rring, phases, k, t=t,
                                            # poyntingstatic=True
                )
        elif general:
            with Timer(self.log.debug, ('dipole_general() (%d points) took '
                       '%%f ms') % (r.shape[0]*r.shape[1])):
                Smean = dipole_general(r, pring, rring, phases, k, poyntingmean=True)
        else:
            t0 = 0
            with Timer(self.log.debug, ('dipole_e_ff() (%d points) took '
                       '%%f ms') % (r.shape[0]*r.shape[1])):
                res = dipole_e_ff(r, pring, rring, phases, k, t=t0)

        if horizontal_line:
            Smag = np.linalg.norm(Smean, axis=2)
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(np.degrees(tlsp), Smag[0, :]/Smag[0, :].max())
        elif sphere:
            if not general:
                ax = self._plot_intens(T, P, res, ax=ax)
            else:
                Smag = np.linalg.norm(Smean, axis=2)
                ax = self._plot_poynting(
                    T, P, Smag, ax=ax,
                    title='poynting vec. on sphere. opening angle $%g^\circ$' % (
                        2*thetamax))
                ax.set_xlabel(r'$\theta_x[^\circ]$', fontsize=15)
                ax.set_ylabel(r'$\theta_y[^\circ]$', fontsize=15)
                # ax = self._plot_poynting(T, P, Smean[:, :, 2], title='poynting on sphere')
                # ax = self._plot_poynting(T, P, res[:, :, 0].real, title='poynting on sphere')
        else:
            if not general:
                # TODO degree x/y label
                ax = self._plot_intens(field=res, XY=(X, Y), ax=ax)
            else:
                Sz = Smean[:, :, 2]
                self.log.debug("Sz min %s, max %s", Sz.min(), Sz.max())
                ax = self._plot_poynting(S=Sz, XY=(X, Y), ax=ax,
                                         title='<Sz> in plane, opening angle %g deg' % (
                                             2*thetamax))
        if not general:
            ax.set_title('evaluation on sphere: %s, opening angle %g deg' % (
                sphere, 2*thetamax))
        self.save_fig('dummy', ax.figure)
        if show:
            self.show()

    def test_fft(self):
        """Here we compare the results of the FFT method (used in the
        papers by Rolf Szedlak) and the exact dipole radiation method.

        In theory there should be a 1:1 correspondence for small angles
        theta between the two methods.

        Rolfs system is a special case as the dipole moments are aligned
        azimuthally and therefore rvec . e_phi is zero, even for large angles
        theta.

        # TODO add unittest: only horizontal pol and vertical pol
        """
        k = 1.
        Lam = 2*np.pi/k
        rringfac = 5
        rring = rringfac*Lam
        incfac = 16
        thetamax = 50.

        self._fft_setup(k, rring, incfac)
        hp, vp, farfield = self._fft_main(k, rring)
        if thetamax is not None:
            farfield[self.theta > np.radians(thetamax)] = np.nan

        # fig, ax = plt.subplots()
        # ax.imshow(hp)
        # fig, ax = plt.subplots()
        # ax.imshow(vp)

        fig, ax = plt.subplots()
        C = ax.pcolormesh(np.degrees(self.tx),
                          np.degrees(self.ty), farfield)
        plt.colorbar(C)
        ax.set_aspect('equal')
        ax.set_xlim(-thetamax, thetamax)
        ax.set_ylim(-thetamax, thetamax)
        self.save_fig('fft_pcolor', fig)

        # TODO splining of datapoints
        fig, ax = plt.subplots()
        rowid = int(self.tx.shape[0]/2)
        xmax = np.nanmax(farfield[rowid, :])
        ax.plot(np.degrees(self.tx[rowid, :]),
                farfield[rowid, :]/xmax, '-x')
        ax.grid(True)
        ax.set_xlabel('theta_x')

        # dipolecode
        self._rolf(sphere=False, shift=False, spp=False, rringfac=rringfac,
                   thetamax=thetamax, horizontal_line=True, show=False, ax=ax, ngrid=1024)
        self.save_fig('fft_line', fig)

        self._rolf(sphere=True, shift=False, spp=False, rringfac=rringfac,
                   thetamax=thetamax, show=False)

        self.show()

    def test_theta_phi_grid(self):
        k = 1.
        Lam = 2*np.pi/k
        rringfac = 5
        rring = rringfac*Lam
        incfac = 2

        self._fft_setup(k, rring, incfac)

        # plot thetax, thetay mesh
        fig, ax = plt.subplots()
        ax.plot(self.tx, self.ty, '-k')
        ax.plot(self.tx.T, self.ty.T, '-k')
        ax.set_aspect('equal')
        self.save_fig('mesh', fig)
        self.show()

    @pytest.mark.parametrize('sphere', [True, False])
    def test_rolf(self, sphere):
        """
        N dipoles on a circle pointing in azimuthal direction
        """
        # TODO eigentlich müsste man annehmen, dass es sich um eine laufende
        # Welle durch den ringqcl handelt.
        # auch wenn eine welle duch das
        # system läuft sollte die phase der dipole zu jedem zeitpunkt immer
        # gleich sein.
        self._rolf(sphere, shift=False, spp=False, thetamax=10)

    @pytest.mark.parametrize('sphere', [True, False])
    def test_rolf_pishift(self, sphere):
        """
        N dipoles on a circle pointing in azimuthal direction with pi shift
        """
        self._rolf(sphere, shift=True, spp=False, thetamax=10)

    @pytest.mark.parametrize('sphere', [True, False])
    def test_rolf_spp(self, sphere):
        """N dipoles on a circle pointing in azimuthal direction with a spiral
        phase element

        """
        self._rolf(sphere, shift=False, spp=True, thetamax=10)

    @pytest.mark.parametrize('sphere', [True, False])
    def test_rolf_spp_custom_ramp(self, sphere):
        """N dipoles on a circle pointing in azimuthal direction with a spiral
        phase element

        """
        evalname = '_eval_on_sphere' if sphere else '_eval_in_plane'
        for ramp_h_fac in (1.2, 1.4, 2.):
            self._rolf(sphere, shift=False, spp=True, thetamax=10,
                       spp_ramp_height=ramp_h_fac*np.pi, show=False)
            plt.gca().set_title(('[ramp height %gpi] ' % ramp_h_fac) +
                                plt.gca().get_title())
            self.save_fig('ramp_height_%gpi%s' % (ramp_h_fac, evalname), plt.gcf())
        self.show()

    @pytest.mark.parametrize('sphere', [True, False])
    def test_rolf_pishift_anim(self, sphere):
        """
        N dipoles on a circle pointing in azimuthal direction with pi shift
        """
        k = 1.  # todo copy and pasted
        import scipy.constants.constants as co
        omega = k*co.c
        tmax = 2*np.pi/omega  # max. time
        shift = False

        ngrid = 256
        La = 2*np.pi/k
        ndip = 1
        z0 = 100.*La
        r = np.empty((ngrid, ngrid, 3))
        thetamax = 15.
        self.log.info('zeval/lambda: %g', z0/La)
        if sphere:
            P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=thetamax,
                                                 ngrid=ngrid)
            self.tx = T*np.cos(P)
            self.ty = T*np.sin(P)
            for i in range(3):
                r[:, :, i] = z0 * e_r[i, :, :]
        else:
            thetamax = min(15., thetamax)
            tm = np.radians(thetamax)
            rmax = z0 * np.tan(tm)
            rng = np.linspace(-rmax, rmax, ngrid)
            X, Y = np.meshgrid(rng, rng)
            r[:, :, 0] = X
            r[:, :, 1] = Y
            r[:, :, 2] = z0

        dipole_phis, dphi = np.linspace(0, 2*np.pi, ndip,
                                        retstep=True, endpoint=False)
        ind = (dipole_phis > np.pi/2) & (dipole_phis < 3*np.pi/2)
        pringfac = 1.
        pring = np.zeros((ndip, 3))  # dipole moments
        ringslice = slice(None, None, None)
        pring[ringslice, 0] = -np.sin(dipole_phis) * pringfac
        pring[ringslice, 1] = np.cos(dipole_phis) * pringfac

        rfac = 15.*La
        rring = np.zeros((ndip, 3))  # dipol aufpunkte
        rring[ringslice, 0] = np.cos(dipole_phis) * rfac
        rring[ringslice, 1] = np.sin(dipole_phis) * rfac

        phases = np.zeros(ndip)
        if shift:
            phases[ind] = np.pi
        with Timer(self.log.debug, ('dipole_general() (%d points) took '
                                    '%%f ms') % (r.shape[0]*r.shape[1])):
            Eres, Hres = dipole_general(r, pring, rring, phases, k, t=0)

        ts = np.linspace(0., tmax, 20)
        # fig, axes = plt.subplots(nrows=len(ts), figsize=(6*len(ts), 5))
        # for t, ax in zip(ts, axes):
        #     fig, ax = plt.subplots()
        #     scur = (0.5*np.cross(Eres,
        #                          Hres.conjugate() + Hres*np.exp(2j*k*co.c*t)).real)
        #     ax.imshow(scur[:, :, 2])
        #     ax.set_ylabel('t=%g' % t)

        # matplotlib animation
        fig, ax = plt.subplots()
        if sphere:
            ax.set_xlim(-thetamax, thetamax)
            ax.set_ylim(-thetamax, thetamax)
            ax.set_xlabel('thetax [deg]')
            ax.set_ylabel('thetay [deg]')
            # ax.set_title('Ex,Ey as a function of phi,theta')
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Ex,Ey as a function of x,y')
            # TODO limits

        def animate(i):
            self.log.info('[%d/%d] ', i+1, len(ts))
            z = 0.5*np.cross(Eres,
                             Hres.conjugate() + Hres*np.exp(2j*k*co.c*ts[i])).real
            # res = dipole_e_ff(r, pring, rring, phases, k=21, t=ts[i])
            z = np.linalg.norm(z, axis=2)
            qvs = ax.contourf(np.degrees(self.tx), np.degrees(self.ty), z, 25)
            ax.set_title('t = %g' % ts[i])
            return qvs

        from matplotlib import animation
        ani = animation.FuncAnimation(fig, animate, frames=len(ts))
        self.show()

    def test_simple_anim(self):
        """
        test matplotlib animation framework
        """
        k = 1.
        import scipy.constants.constants as co
        omega = k*co.c
        tmax = 2*np.pi/omega  # max. time
        ngrid = 256
        thetamax = 15.
        P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=thetamax,
                                             ngrid=ngrid)
        self.tx = T*np.cos(P)
        self.ty = T*np.sin(P)

        ts = np.linspace(0., tmax, 20)
        fig, ax = plt.subplots()
        ax.set_xlim(-thetamax, thetamax)
        ax.set_ylim(-thetamax, thetamax)
        ax.set_xlabel('thetax [deg]')
        ax.set_ylabel('thetay [deg]')

        def animate(i):
            self.log.info('[%d/%d] ', i+1, len(ts))
            qvs = ax.contourf(np.degrees(self.tx), np.degrees(self.ty),
                              self.tx + i**0.2*self.ty)
            ax.set_title('t = %g' % ts[i])
            return qvs

        from matplotlib import animation
        ani = animation.FuncAnimation(fig, animate, frames=len(ts))
        # ani.save('test.mp4')
        self.show()

    def test_gaussian(self):
        # TODO add more logic to this test
        from dipole.utils import GaussianBeam
        gb = GaussianBeam()
        gb.k = 1.
        gb.z = 0
        gb.w0 = 1.
        gb.E0 = 1.

        ngrid = 128
        z0 = 100.
        thetamax = 15.
        tm = np.radians(thetamax)
        hyp = z0 / np.cos(tm)
        rmax = hyp * np.sin(tm)
        rng = np.linspace(-rmax, rmax, ngrid)
        X, Y = np.meshgrid(rng, rng)
        R = np.hypot(X, Y)

        Efield = gb.eval(0, R)

        fig, ax = plt.subplots()
        ax.contourf(X, Y, np.angle(Efield))
        fig, ax = plt.subplots()
        ax.contourf(X, Y, abs(Efield)**2)
        self.show()
