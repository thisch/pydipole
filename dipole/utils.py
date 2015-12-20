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


# WIP
class GaussianBeam:

    def waist(self, z):
        return 25.

    def R(self, z):
        return 1.

    def gouy(self, z):
        return 0.

    def eval(self, z, r):
        import numpy as np
        return self.E0*self.w0/self.waist(z)*np.exp(
            -r**2/self.waist(z)**2 +
            -1j*self.k*z +
            -1j*self.k*r**2/(2*self.R(z)) +
            -1j*self.gouy(z))
