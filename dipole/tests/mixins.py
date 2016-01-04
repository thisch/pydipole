import numpy as np

from ..utils import Timer


class FFTMixin:
    def _fourier_single(self, data):
        with Timer(self.log.debug, 'single fft: %f ms'):
            absfft = abs(np.fft.fftshift(np.fft.fft2(data, s=self.tx.shape)))
        return absfft**2

    def _fourier_hvp(self, hp, vp):
        with Timer(self.log.debug, '2x fft: %f ms'):
            absffthp = abs(np.fft.fftshift(np.fft.fft2(hp, s=self.tx.shape)))
            absfftvp = abs(np.fft.fftshift(np.fft.fft2(vp, s=self.tx.shape)))
            farfield = absffthp**2 + absfftvp**2
        return farfield

    def _fft_setup(self, k, rmax, incfac, rextfac=1.05):
        with Timer(self.log.debug, "_fft_setup took %f ms"):
            xlinsp = np.linspace(-rextfac*rmax, rextfac*rmax, 128*2)
            ylinsp = np.linspace(-rextfac*rmax, rextfac*rmax, 128*2)
            X, Y = np.meshgrid(xlinsp, ylinsp)

            kx = 2*np.pi*np.fft.fftfreq(incfac*xlinsp.size,
                                        d=(xlinsp[1]-xlinsp[0]))
            ky = 2*np.pi*np.fft.fftfreq(incfac*ylinsp.size,
                                        d=(ylinsp[1]-ylinsp[0]))
            kX, kY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))
            self.log.debug('[fft_setup] FFT shape %s', str(kX.shape))
            krho = np.hypot(kX, kY)
            sqrddiff = k**2 - kX**2 - kY**2
            sqrddiff[krho > k] = 0.
            kZ = np.sqrt(sqrddiff)
            del sqrddiff

            theta = np.pi/2. - np.angle(krho + 1j*kZ)
            phi = np.angle(kX + 1j*kY)
            tx, ty = theta*np.cos(phi), theta*np.sin(phi)
            self.theta = theta
            self.tx, self.ty = tx, ty
            self.X, self.Y = X, Y

    def _fft_main(self, k, rmax, hvpfunc=None):
        if hvpfunc is not None:
            with Timer(self.log.debug, 'hvpfunc took %f ms'):
                hp, vp = hvpfunc(self.X, self.Y)
        else:
            Lam = 2*np.pi/k
            R = np.hypot(self.X, self.Y)
            P = np.angle(self.X + 1j*self.Y)
            delta = 0.05
            rlow, rup = rmax - delta*Lam, rmax + delta*Lam

            hp = np.zeros_like(R)
            vp = np.zeros_like(R)
            ind = (R < rup) & (R > rlow)
            hp[ind] = (-np.sin(P[ind]))
            vp[ind] = (np.cos(P[ind]))

        farfield = self._fourier_hvp(hp, vp)
        return hp, vp, farfield


class DipoleMixin:
    # TODO
    pass
