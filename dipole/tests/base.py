import pytest
import os
import numpy as np
import logging
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Base:
    log = logging.getLogger('dip')

    def setup_method(self, method):
        print("\n{}:{}".format(self.__class__.__name__, method.__name__))
        # TODO support for fixtures (convert fixtures to strs)
        self._testMethodName = method.__name__

    def _plot_intens(self, T=None, P=None, field=None, title=None, XY=None, ax=None):
        intens = np.sum(np.abs(field)**2, axis=2)
        if ax is None:
            fig, ax = plt.subplots()
        # ax = fig.gca(projection='3d')

        # ax.plot_surface(intens*r[:, :, 0],
        #                 intens*r[:, :, 1],
        #                 intens*r[:, :, 2])
        # ax.contourf(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
        #             intens)

        # ax.imshow(intens)
        if XY is not None:
            ax.pcolormesh(XY[0], XY[1], intens)
        else:
            ax.pcolormesh(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)),
                          intens)
        # TODO improve limits (take code from paper_plotter.py)

        ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        # ax.set_xlim(-25,25)
        # ax.set_ylim(-25,25)

        # cax.set_xticks([])
        # cax.set_yticks([])
        # cax.set_title('R_disk=%g lambda=%g, ndipoles=%d' % (scalefac, 2*np.pi/k,
                                                            # ndipoles))
        return ax

    def _plot_poynting(self, T=None, P=None, S=None, title=None, XY=None, ax=None):
        """s is either the magnitude of the poynting vector or a component of the
        poynting vector

        """
        if ax is None:
            fig, ax = plt.subplots()
        if XY is not None:
            C = ax.pcolormesh(XY[0], XY[1], S)
        else:
            C = ax.pcolormesh(np.degrees(T*np.cos(P)), np.degrees(T*np.sin(P)), S)
        if title:
            ax.set_title(title)
        ax.set_aspect('equal')

        plt.colorbar(C)

        return ax

    def gen_filename(self, postfix, extension):
        fname = "%s_%s%s.%s" % (self.__class__.__name__,
                                self._testMethodName,
                                "_%s" % postfix if postfix else "",
                                extension)
        logd = pytest.config.getvalue('logdir')
        fname = os.path.join(logd, fname)
        return fname

    def save_fig(self, postfix, fig, ext='png'):
        assert not postfix.endswith('.png')
        dest = self.gen_filename(postfix, ext)
        if not pytest.config.getvalue('log'):
            self.log.debug("do not save png %s (--log not set)", dest)
            return
        self.log.debug("saving mpl figure to '%s'", dest)
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        fig.savefig(dest)
        # TODO
        # self.created_files.append(dest)

    def show(self):
        if pytest.config.getvalue('interactive'):
            plt.show()
