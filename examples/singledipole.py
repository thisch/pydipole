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
    else:
        ax.pcolormesh(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
                      intens)
        ax.set_xlabel(r'$\Theta_x$', fontsize=16)
        ax.set_ylabel(r'$\Theta_y$', fontsize=16)
        tm = 90
        ax.set_xlim(-tm, tm)
        ax.set_ylim(-tm, tm)
        ax.set_xticks([-tm, -45, 0, 45, tm])
        ax.set_yticks([-tm, -45, 0, 45, tm])
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    return ax


def main(onsphere=False):
    if not onsphere:
        thetamax = 40.
    else:
        thetamax = 90.
    k = 1.
    Lam = 2*np.pi/k
    reval = 1000*Lam
    ngrid = 128

    for align_axis in 'xyz':
        LG.info('#### SETTINGS: k=%g, reval=%g', k, reval)
        rparams = gen_r(ngrid, reval=reval, onsphere=onsphere, thetamax=thetamax)

        pdisk = np.zeros((1, 3))
        if align_axis == 'z':
            pdisk[:, 2] = 1.
        elif align_axis == 'x':
            pdisk[:, 0] = 1.
        elif align_axis == 'y':
                    pdisk[:, 1] = 1.

        rdip = np.zeros((1, 3))
        phases = np.zeros(1)

        if onsphere:
            # radiant intensity
            intens = dipole_radiant_intensity(rparams[0],
                                              rparams[1],
                                              pdisk, rdip, phases, k)
        else:
            # is it better to calculate the intensity in z-direction and not
            # the radiant-intensity?
            eff = dipole_e_ff(rparams[-1], pdisk, rdip, phases, k, t=0)
            intens = np.linalg.norm(eff, axis=2)**2
        if onsphere:
            T, P, _ = rparams
            ax = plot_intens(T, P, intens)
        else:
            X, Y, _, = rparams
            ax = plot_intens(intens=intens, XY=(X, Y))
        ax.set_title('k=%g, dipole orientation: %s-axis' % (k, align_axis))
    plt.show()


if __name__ == '__main__':
    main(onsphere=True)
    main(onsphere=False)
