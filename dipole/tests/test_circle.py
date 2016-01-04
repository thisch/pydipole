import numpy as np

from ..helper import unit_vectors
from ..field import dipole_e_ff
from ..utils import Timer

from .base import Base
from .mixins import FFTMixin

import matplotlib.pyplot as plt


class TestCircle(Base, FFTMixin):

    def analytic_rad(self, k, rcirc, krfac=0.4):
        """
        calculated the far-field diffraction profile of a plane wave incident on
        a circular aperture, (Airy function)

        Returns:
        --------
        xdata: array (theta (deg))
        ydata: array (intensity)

        """
        from scipy.special import j1
        krmax = k*krfac
        kr = np.linspace(-krmax, krmax, 256)  # = +-hypot(kx, ky)
        kz = np.sqrt(k**2 - kr**2)
        theta = np.angle(kz + 1j*kr)

        # # see Fourier scaling theorem
        # scaling = abs(rcirc)
        # krscal = kr*rcirc
        # Fcirc = scaling * j1(krscal)*2*np.pi/krscal

        # expression of the airy pattern (see wikipedia Airy Disk)
        krscal = rcirc*kr  # = k*rcirc*np.sin(theta)
        Fcirc = 2*j1(krscal)/krscal

        xdata = np.degrees(theta)  # theta in deg
        ydata = (Fcirc/Fcirc.max())**2  # normalized intensity
        return xdata, ydata

    def test_fft(self):
        """
        calculate the far-field diffraction profile numerically using FFT
        """
        k = 0.08
        rcirc = 250.
        incfac = 16
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

        # TODO splining of datapoints
        fig, ax = plt.subplots()
        rowid = int(self.tx.shape[0]/2)
        ff = farfield[rowid, :]
        xdata = np.degrees(self.tx[rowid, :])
        ydata = ff/np.nanmax(ff)  # farfield data normalized to 1
        ax.plot(xdata, ydata, '-x', label='fft')
        ax.plot(*self.analytic_rad(k, rcirc), ls='-', c='r', label='analytic')
        ythresh = np.exp(-2)
        ax.axhline(y=ythresh, c='k')
        ax.grid(True)
        beamdiv = xdata[ydata > ythresh].ptp()
        # analytical beam divergence angle:
        ax.set_title('numerical divergence angle: %g°' % beamdiv)
        ax.set_xlabel('theta_x')
        ax.legend(fontsize=8)
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
        numdps = 4000
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

        ngrid = 1024
        r = np.empty((1, ngrid, 3))
        tlsp = np.radians(np.linspace(-thetamax, thetamax, ngrid))
        r[0, :, 0] = reval * np.sin(tlsp)
        r[0, :, 1] = 0
        r[0, :, 2] = reval * np.cos(tlsp)
        with Timer(self.log.debug,
                   'dipole_e_ff (ndip: %d) at %d eval pts took %%f ms' % (
                       numdps, ngrid)):
            res = dipole_e_ff(r, pdp, rdip, dipole_phis, k, t=0)
        farfield = (np.linalg.norm(res, axis=2)**2).flatten()
        farfield /= farfield.max()
        plt.subplots()

        plt.plot(np.degrees(tlsp), farfield, '-', label='dipole rad')
        plt.plot(*self.analytic_rad(k, rcirc), ls='-', c='r',
                 label='analytic aperture rad')
        thline = np.exp(-2)
        plt.axhline(y=thline, c='k')
        twotheta = tlsp[farfield > thline].ptp()
        plt.title("2*theta = %g deg" % (np.degrees(twotheta)))
        plt.grid(True)
        plt.legend(fontsize=8)
        self.show()

    def test_analytic_ft(self):
        """
        far-field diffraction of a plane wave incident on a circular aperture
        """
        k = 0.08
        rcirc = 250.
        xdata, ydata = self.analytic_rad(k, rcirc)

        fig, ax = plt.subplots()
        ax.plot(xdata, ydata, '-x')
        ax.set_xlabel(r'$\theta [°]$')
        ax.set_ylabel(r'|FT(uniform disk)|^2')
        ythresh = np.exp(-2)
        ax.axhline(y=ythresh, c='k')
        ax.grid(True)
        beamdiv = xdata[ydata > ythresh].ptp()
        ax.set_title('analytical divergence angle: %g°' % beamdiv)
        assert beamdiv < 14.6
        assert beamdiv > 14.0
        self.show()
