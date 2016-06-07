import matplotlib.pyplot as plt
import numpy as np


def plot_linpol(T, P, field, fixvminmax=True, ax=None):
    # radial unit vec
    # e_r = np.zeros(P.shape + (3,))
    # e_r[:, :, 0] = np.sin(T)*np.cos(P)
    # e_r[:, :, 1] = np.sin(T)*np.sin(P)
    # e_r[:, :, 2] = np.cos(T)

    # theta unit vec
    e_t = np.zeros(P.shape + (3,))
    e_t[:, :, 0] = np.cos(T)*np.cos(P)
    e_t[:, :, 1] = np.cos(T)*np.sin(P)
    e_t[:, :, 2] = -np.sin(T)

    # phi unit vec
    e_p = np.zeros(P.shape + (3,))
    e_p[:, :, 0] = -np.sin(P)
    e_p[:, :, 1] = np.cos(P)

    # Er = sum([field[:, :, i]*e_r[:, :, i] for i in range(3)])
    Et = sum([field[:, :, i]*e_t[:, :, i] for i in range(3)])
    Ep = sum([field[:, :, i]*e_p[:, :, i] for i in range(3)])

    def intcalc(f):
        # returns  < (f.real(t))^2 >
        return np.abs(f)**2

    allint = np.sum(intcalc(field), axis=2)
    norm = allint.max()
    S0 = (intcalc(Et) + intcalc(Ep))/norm

    # checks that we are in the far-field
    import numpy.testing as nt
    nt.assert_allclose(allint/norm, S0, atol=1e-10)

    S1 = (intcalc(Et) - intcalc(Ep))/norm

    # fig, ax = plt.subplots()
    # cax = ax.pcolormesh(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
    #                     intcalc(Er)/norm)
    # plt.colorbar(cax)
    # ax.set_title("abs Er")

    # fig, ax = plt.subplots()
    # cax = ax.pcolormesh(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
    #                     (intcalc(Et) + intcalc(Ep))/norm)
    # plt.colorbar(cax)
    # ax.set_title("abs Et + abs Ep")
    # self.show()

    Ep45 = np.cos(np.radians(45))*Et + np.sin(np.radians(45))*Ep
    Epm45 = np.cos(np.radians(-45))*Et + np.sin(np.radians(-45))*Ep
    S2 = (intcalc(Ep45) - intcalc(Epm45))/norm

    Pi = np.sqrt(S1**2 + S2**2)/S0

    if ax is None:
        fig, ax = plt.subplots()
    if fixvminmax:
        kwargs = dict(vmin=0, vmax=1)
    else:
        kwargs = {}
    cax = ax.pcolormesh(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
                        Pi, **kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('theta_x [deg]')
    ax.set_ylabel('theta_y [deg]')
    plt.colorbar(cax, ax=ax, orientation='horizontal')


def plot_linpol_plane(X, Y, field, fixvminmax=True, ax=None):

    def intcalc(f):
        # returns  < (f.real(t))^2 >
        return np.abs(f)**2

    Ex, Ey = field[:, :, 0], field[:, :, 1]
    allint = np.sum(intcalc(field), axis=2)
    norm = allint.max()

    # or allint?
    S0 = (intcalc(Ex) + intcalc(Ey))/norm

    S1 = (intcalc(Ex) - intcalc(Ey))/norm

    Ep45 = np.cos(np.radians(45))*Ex + np.sin(np.radians(45))*Ey
    Epm45 = np.cos(np.radians(-45))*Ex + np.sin(np.radians(-45))*Ey
    S2 = (intcalc(Ep45) - intcalc(Epm45))/norm

    Pi = np.sqrt(S1**2 + S2**2)/S0

    if ax is None:
        fig, ax = plt.subplots()
    if fixvminmax:
        kwargs = dict(vmin=0, vmax=1)
    else:
        kwargs = {}
    cax = ax.pcolormesh(X, Y, Pi, **kwargs)
    ax.set_aspect('equal')
    plt.colorbar(cax)
