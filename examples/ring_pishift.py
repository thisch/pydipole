#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import logging

LG = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

from dipole.field import dipole_e_ff
from dipole.field import dipole_radiant_intensity
from dipole.helper import gen_r


def plot_intens(T=None, P=None, intens=None, title=None, XY=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if XY is not None:
        ax.pcolormesh(XY[0], XY[1], intens)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
    else:
        ax.pcolormesh(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
                      intens)
        ax.set_xlabel(r'$\Theta_x$', fontsize=16)
        ax.set_ylabel(r'$\Theta_y$', fontsize=16)
        tm = 10
        ax.set_xlim(-tm, tm)
        ax.set_ylim(-tm, tm)
        # ax.set_xticks([-tm, -45, 0, 45, tm])
        # ax.set_yticks([-tm, -45, 0, 45, tm])
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    return ax


def main(onsphere=False):
    thetamax = 10.
    k = 1.
    Lam = 2*np.pi/k
    reval = 1000*Lam
    ngrid = 128
    ndip = 256
    reval = 500.*Lam

    LG.info('#### SETTINGS: k=%g, reval=%g', k, reval)
    rparams = gen_r(ngrid, reval=reval, onsphere=onsphere, thetamax=thetamax)

    dipole_phis, dphi = np.linspace(0, 2*np.pi, ndip,
                                    retstep=True, endpoint=False)
    pringfac = 1.
    pring = np.zeros((ndip, 3))  # dipole moments
    ringslice = slice(None, None, None)
    pring[ringslice, 0] = -np.sin(dipole_phis) * pringfac
    pring[ringslice, 1] = np.cos(dipole_phis) * pringfac

    rringfac = 15.
    rfac = rringfac*Lam
    rring = np.zeros((ndip, 3))  # dipol aufpunkte
    rring[ringslice, 0] = np.cos(dipole_phis) * rfac
    rring[ringslice, 1] = np.sin(dipole_phis) * rfac

    phases = np.zeros(ndip)

    # set that phase of all dipoles with x < 0 to pi
    ind = (dipole_phis > np.pi/2) & (dipole_phis < 3*np.pi/2)
    phases[ind] = np.pi

    if onsphere:
        # radiant intensity
        intens = dipole_radiant_intensity(rparams[0],
                                          rparams[1],
                                          pring, rring, phases, k)
    else:
        eff = dipole_e_ff(rparams[-1], pring, rring, phases, k, t=0)
        intens = np.linalg.norm(eff, axis=2)**2
    if onsphere:
        T, P, _ = rparams
        ax = plot_intens(T, P, intens)
    else:
        X, Y, _, = rparams
        ax = plot_intens(intens=intens, XY=(X, Y))
    ax.set_title('k=%g, %s' % (k,
                               ('reval=%g' % reval) if onsphere else
                               ('zeval=%g' % reval)))



if __name__ == '__main__':
    main(onsphere=True)
    main(onsphere=False)
    plt.show()
