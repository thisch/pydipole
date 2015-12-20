import numpy as np


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
