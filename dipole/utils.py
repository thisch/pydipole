import numpy as np

from timeit import default_timer


class Timer(object):
    def __init__(self, stream, msg=None):
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


class GaussianBeam:

    def __init__(self, waist_size, k, E0, pol):
        self.w0 = waist_size
        self.k = k
        self.E0 = E0
        Lam = 2*np.pi/k
        self.zr = np.pi*self.w0**2/Lam  # Rayleigh range
        self.pol = pol

    def q(self, z):
        """
        complex beam parameter
        """
        return z + 1j*self.zr

    def qinv(self, z):
        """
        inverse of the complex beam parameter
        """
        return 1./self.R(z) - 2j/(self.k*self.w(z)**2)

    def w(self, z):
        """radius at which the field amplitudes fall to 1/e of their axial values,
        at the plane z along the beam,

        """
        return self.w0*np.sqrt(1 + (z/self.zr)**2)

    def R(self, z):
        """
        Radius of curvature
        """
        return z*(1 + (self.zr/z)**2)

    def gouy(self, z):
        return np.arctan(z/self.zr)

    def eval(self, z, r):
        scalval = 1/self.q(z)*np.exp(-1j*self.k*r**2/(2*self.q(z)))
        return self.add_polarization(scalval, z, r)

    def eval_vec(self, rvec):
        assert rvec.ndim == 3
        z = rvec[:, :, 2]
        r = np.hypot(rvec[:, :, 0], rvec[:, :, 1])
        return self.eval(z, r)

    def add_polarization(self, scalar, z, r):
        ones = np.ones_like(scalar)
        # the following expressions are valid for the exp(-iwt) time
        # dependence of the phasors.
        if self.pol == 'LHCP':
            pre = [1., 1j, 0]
        elif self.pol == 'RHCP':
            pre = [1., -1j, 0]
        elif self.pol == 'x':
            pre = [1., 0, 0]
        elif self.pol == 'y':
            pre = [0, 1, 0]
        else:
            raise ValueError('invalid polarization value %s' % self.pol)
        tmp = np.dstack([pre[0]*ones, pre[1]*ones, pre[2]*ones])
        return scalar[:, :, None] * tmp
