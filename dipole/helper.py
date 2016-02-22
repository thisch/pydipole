import logging
import cmath

import numpy as np

LG = logging.getLogger('pyd.hp')


def unit_vectors(thetamax=7.5, ngrid=32):
    thetamax = np.radians(thetamax)
    pls = np.linspace(0, 2*np.pi, ngrid)
    tls = np.linspace(0, thetamax, ngrid)
    P, T = np.meshgrid(pls, tls)

    # Raux = np.hypot(P, T - np.pi / 2)

    # radial unit vec
    e_r = np.zeros((3,) + P.shape)
    e_r[0, :, :] = np.sin(T)*np.cos(P)
    e_r[1, :, :] = np.sin(T)*np.sin(P)
    e_r[2, :, :] = np.cos(T)

    # theta unit vec
    e_t = np.zeros((3,) + P.shape)
    e_t[0, :, :] = np.cos(T)*np.cos(P)
    e_t[1, :, :] = np.cos(T)*np.sin(P)
    e_t[2, :, :] = -np.sin(T)

    # phi unit vec
    e_p = np.zeros((3,) + P.shape)
    e_p[0, :, :] = -np.sin(P)
    e_p[1, :, :] = np.cos(P)

    return P, T, (e_r, e_t, e_p)


def gen_r(ngrid, reval, onsphere, thetamax=None, rmax=None):
    # TODO generalize this function

    r = np.empty((ngrid, ngrid, 3))
    if onsphere:
        assert thetamax is not None
        P, T, (e_r, e_t, e_p) = unit_vectors(thetamax=thetamax,
                                             ngrid=ngrid)
        r[:, :, 0] = reval * e_r[0, :, :]
        r[:, :, 1] = reval * e_r[1, :, :]
        r[:, :, 2] = reval * e_r[2, :, :]
    else:
        if thetamax is not None:
            rmax = np.tan(np.radians(thetamax)) * reval
        assert rmax is not None
        LG.info(" %s deg", np.degrees(cmath.phase(reval + 1j*np.sqrt(2)*rmax)))
        rng = np.linspace(-rmax, rmax, ngrid)
        X, Y = np.meshgrid(rng, rng)
        r[:, :, 0] = X
        r[:, :, 1] = Y
        r[:, :, 2] = reval
    LG.debug("onsphere: %s\tfirst r vec %s", onsphere, r[0, 0, :])
    if onsphere:
        return T, P, r
    else:
        return X, Y, r
