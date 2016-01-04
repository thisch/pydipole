import numpy as np
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from numpy.testing import assert_almost_equal

from ..helper import unit_vectors
from ..field import dipole_e_ff
from ..utils import Timer

from .base import Base
from .mixins import FFTMixin

import matplotlib.pyplot as plt


class TestCircle(Base, FFTMixin):

    def analytic_rad(self, k, rcirc, thetas):
        """
        calculates the far-field diffraction profile of a plane wave incident on
        a circular aperture, (Airy function)

        Parameters:
        -----------
        k: wave number
        rcirc: radius of aperture
        thetas: array of angles [in deg], where the far-field should be evaluated

        Returns:
        --------
        xdata: array (theta (deg))
        ydata: array (intensity)

        """
        from scipy.special import j1

        # expression of the airy pattern (see wikipedia Airy Disk)
        krscal = k*rcirc*np.sin(np.radians(thetas))
        Fcirc = 2*j1(krscal)/krscal
        return thetas, (Fcirc/Fcirc.max())**2  # normalized intensity

    def normalize_and_beamdiv(self, thetas, farfield):
        """
        Parameters:
        -----------
        thetas: array [deg]
        farfield: array containing intensities

        Returns:
        beamdiv: float [deg]
        farfield: normalized farfield intensities
        """
        # cubic interp. is slow :(
        with Timer(self.log.debug, 'interps took %f ms'):
            func = interp1d(thetas, farfield, kind='cubic')
        fmax = func(0)  # we assume that the max is at the center
        farfield = farfield/fmax
        thresh = np.exp(-2)
        with Timer(self.log.debug, 'bisecting took %f ms'):
            # ftol = 1e-10
            rootfunc = lambda x: func(x)/fmax - thresh
            tright = bisect(rootfunc, 0, thetas.max())  # , ftol=ftol)
            tleft = bisect(rootfunc, thetas.min(), 0)  # , ftol=ftol)
            fright, fleft = rootfunc([tright, tleft])
            self.log.debug("debug errl: %g errr: %g" % (abs(fleft), abs(fright)))
        trl = tright - tleft  # beam div.
        self.log.debug('tleft %g, tright %g', tleft, tright)
        return trl, farfield

    def airy_beam_divergence(self, k, r):
        """
        analytical width of an airy disk
        """
        # j1(x) = x/(2*e) -> xana
        xana = 2.58383899
        # xana = k*rcirc*np.sin(np.radians(tana))
        tana = np.degrees(np.arcsin(xana/(k*r)))
        return 2*tana

    def test_fft(self):
        """
        calculate the far-field diffraction profile numerically using FFT
        """
        k = 0.08
        rcirc = 250.
        incfac = 8
        thetamax = 20.

        self._fft_setup(k, rcirc, incfac)

        R = np.hypot(self.X, self.Y)
        nfdata = np.zeros_like(R)
        ind = R < rcirc
        nfdata[ind] = 1.
        farfield = self._fourier_single(nfdata)

        if thetamax is not None:
            farfield[self.theta > np.radians(thetamax)] = np.nan

        fig, ax = plt.subplots()
        C = ax.pcolormesh(np.degrees(self.tx),
                          np.degrees(self.ty), farfield)
        plt.colorbar(C)
        ax.set_aspect('equal')
        ax.set_xlim(-thetamax, thetamax)
        ax.set_ylim(-thetamax, thetamax)
        self.save_fig('fft_pcolor', fig)

        fig, ax = plt.subplots()
        ax.plot(*self.analytic_rad(k, rcirc,
                                   np.linspace(-thetamax, thetamax, 1024)),
                ls='-', c='r', label='analytic')

        rowid = int(self.tx.shape[0]/2)
        xdata = np.degrees(self.tx[rowid, :])
        ydata = farfield[rowid, :]
        # remove nan values
        ind = ~(np.isnan(xdata) | np.isnan(ydata))
        xdata, ydata = xdata[ind], ydata[ind]
        beamdiv, ydata = self.normalize_and_beamdiv(xdata, ydata)
        ax.plot(xdata, ydata, '-x', label='fft')

        airydiv = self.airy_beam_divergence(k, rcirc)
        ax.axhline(y=np.exp(-2), c='k')
        ax.grid(True)
        ax.set_title('numerical divergence angle: %g° (airy disk width %g°)' % (
            beamdiv, airydiv))
        ax.set_xlabel('theta_x')
        ax.legend(fontsize=8)
        assert_almost_equal(beamdiv, airydiv, decimal=2)
        self.show()

    def test_dipole_density(self):
        """
        radiation profile of randomly distributed dipoles with a constant phase
        and direction/polarization.

        """
        k = 0.08
        Lam = 2*np.pi/k
        reval = 1000*Lam
        rcirc = 250.
        numdps = 1000
        thetamax = 30

        rdp = np.random.rand(numdps, 2)
        rdp = (rdp - 0.5)*2*rcirc
        rdp = rdp[rdp[:, 0]**2 + rdp[:, 1]**2 <= rcirc**2]
        # rdp = rdp[:1, :]  # 1 dp
        numdps = rdp.shape[0]  # actual number of dps inside a circle

        ngrid = 64
        P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=thetamax, ngrid=ngrid)
        tx, ty = T*np.cos(P), T*np.sin(P)

        r = np.empty((ngrid, ngrid, 3))
        r[:, :, 0] = reval * e_r[0, :, :]
        r[:, :, 1] = reval * e_r[1, :, :]
        r[:, :, 2] = reval * e_r[2, :, :]
        self.log.debug("first r vec %s", r[0, 0, :])

        pdp = np.zeros((numdps, 3))  # dipole moments
        pdp[:, 0] = 1  # px != 0
        dipole_phis = np.zeros(numdps)

        rdip = np.empty((numdps, 3))
        rdip[:, 0] = rdp[:, 0]
        rdip[:, 1] = rdp[:, 1]
        rdip[:, 2] = 0.  # all dipoles lay in the xy-plane
        del rdp

        with Timer(self.log.debug,
                   'dipole_e_ff (ndip: %d) took %%f ms' % numdps):
            res = dipole_e_ff(r, pdp, rdip, dipole_phis, k, t=0)
        farfield = np.linalg.norm(res, axis=2)**2

        plt.subplots()
        plt.title('spatial dipole configuration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(rdip[:, 0], rdip[:, 1])

        plt.subplots()
        plt.contourf(np.degrees(tx), np.degrees(ty), farfield)
        plt.xlabel('theta_x [deg]')
        plt.ylabel('theta_y [deg]')

        phis = np.linspace(0, 2*np.pi, 16, endpoint=False)
        phis.shape = (1, -1)
        ngrid = 512
        r = np.empty((1, ngrid*phis.size, 3))
        tlsp = np.linspace(-thetamax, thetamax, ngrid)
        tlsprad = np.radians(tlsp)
        tlsprad.shape = (-1, 1)
        r[0, :, 0] = (reval * np.sin(tlsprad) * np.cos(phis)).flatten()
        r[0, :, 1] = (reval * np.sin(tlsprad) * np.sin(phis)).flatten()
        r[0, :, 2] = (reval * np.cos(tlsprad) * np.ones_like(phis)).flatten()
        with Timer(self.log.debug,
                   'dipole_e_ff (ndip: %d) at %d eval pts took %%f ms' % (
                       numdps, ngrid)):
            res = dipole_e_ff(r, pdp, rdip, dipole_phis, k, t=0)
        allfarfield = (np.linalg.norm(res, axis=2)**2).flatten()

        # TODO determine 1/e^2 contour and plot this contour in a r(phi) plot
        # and in a theta_y theta_x plot.
        divs = []
        for i in range(phis.size):
            farfield = allfarfield[i::phis.size]
            beamdiv, farfield = self.normalize_and_beamdiv(tlsp, farfield)
            self.log.info("phi={} deg: beamdiv: {} deg".format(
                np.degrees(phis.flat[i]), beamdiv))
            divs.append(beamdiv)
            plt.subplots()
            plt.plot(tlsp, farfield, '-', label='dipole rad')
            plt.plot(*self.analytic_rad(k, rcirc, tlsp),
                     ls='-', c='r', label='analytic aperture rad')
            plt.axhline(y=np.exp(-2), c='k')
            plt.title("phi=%g deg, beamdiv = %g deg" % (np.degrees(phis.flat[i]), beamdiv))
            plt.grid(True)
            plt.legend(fontsize=8)
        divs = np.array(divs)
        self.log.info('beamdivergences: min {}, max {}, mean {}, std {}'.format(
            divs.min(), divs.max(), divs.mean(), divs.std()))

        plt.subplots()
        plt.title('beam divergences')
        plt.plot(np.degrees(phis.flatten()), divs, 'o')
        plt.grid(True)
        plt.xlabel('phi')

        plt.subplots()
        plt.title('beam shape')
        pfine = np.linspace(0, 2*np.pi, 512)
        divm = divs.mean()
        plt.plot(divs*np.cos(phis.flatten()),
                 divs*np.sin(phis.flatten()), 'o-')
        plt.plot(divm*np.cos(pfine), divm*np.sin(pfine), 'r-', alpha=0.5)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')

        self.show()

    def test_analytic_ft(self):
        """
        far-field diffraction of a plane wave incident on a circular aperture
        """
        k = 0.08
        rcirc = 250.
        tm = 40
        thetas = np.linspace(-tm, tm, 1024)
        xdata, ydata = self.analytic_rad(k, rcirc, thetas)

        self.log.debug('2*tana %g deg', self.airy_beam_divergence(k, rcirc))

        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\theta [°]$')
        ax.set_ylabel(r'|FT(uniform disk)|^2')
        ythresh = np.exp(-2)
        ax.axhline(y=ythresh, c='k')
        ax.grid(True)

        # TODO std. of beamdiv
        beamdiv, ydata = self.normalize_and_beamdiv(thetas, ydata)
        ax.plot(xdata, ydata, '-')
        ax.axvspan(-beamdiv/2., beamdiv/2., color='r', alpha=0.5)
        ax.set_title('divergence angle: %g°' % beamdiv)
        assert_almost_equal(beamdiv, self.airy_beam_divergence(k, rcirc), decimal=4)
        self.show()
