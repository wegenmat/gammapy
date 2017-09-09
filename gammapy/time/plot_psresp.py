import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
rc('text', usetex=True)
rc('font', size=24)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def plot_psresp(slopes, binning, df, suf, best_parameters):
    """
    Plot the success fraction over slopes for parameters satisfying the significance criterion
    and the histogram over the grid of parameters.

    Parameters
    ----------
    slopes : `~numpy.ndarray`
        slopes of the power law model
    binning : `~numpy.ndarray`
        bin length for the light curve in units of ``t``
    df : `~numpy.ndarray`
        bin factor for the logarithmic periodogram
    suf : `~numpy.ndarray`
        Success fraction for each model parameter
    best_parameters : `~numpy.ndarray`
        Parameters satisfying the significance criterion

    Returns
    -------
    plot
    """

    fig = plt.figure(figsize=(11, 11))
    for indx in range(len(best_parameters[0])):
        plt.plot(-slopes, suf[:, binning == binning[best_parameters[0][indx]], df == df[best_parameters[1][indx]]],
                 label='$ \Delta t_{{bin}} = {} $, $ \Delta f = {} $'.format(binning[best_parameters[0][indx]],
                                                                             df[best_parameters[1][indx]]))
    plt.xlabel(r'\textbf{slope}')
    plt.ylabel(r'\textbf{success fraction}')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('SuF', bbox_inches='tight')


    fig = plt.figure(figsize=(11, 11))
    ax = fig.gca(projection='3d')

    X, Y, Z = -slopes, binning, df

    suf_test = suf > 0.95 * np.max(suf)
    sumz = np.sum(suf_test, axis=2)
    x, y = np.meshgrid(X, Y)
    xy = ax.contour(x, y, sumz.T, zdir='z', offset=0)

    sumy = np.sum(suf_test, axis=1)
    x, z = np.meshgrid(X, Z)
    xz = ax.contour(x, sumy.T, z, zdir='y', offset=7)

    sumx = np.sum(suf_test, axis=0)
    y, z = np.meshgrid(Y, Z)
    yz = ax.contour(sumx.T, y, z, zdir='x', offset=0.9)

    ax.set_xlim([0.9, 2.6])
    ax.set_ylim([1, 7])
    ax.set_zlim([0, 1.2])

    ax.set_xlabel(r'$slope$', labelpad=20)
    ax.set_ylabel(r'$\Delta t$', labelpad=20)
    ax.set_zlabel(r'$\Delta f$', labelpad=20)

    cxy = plt.colorbar(xy)
    cxy.ax.set_title(r'$slope$')
    cxz = plt.colorbar(xz)
    cxz.ax.set_title(r'$\Delta t$')
    cyz = plt.colorbar(yz)
    cyz.ax.set_title(r'$\Delta f$')

    plt.savefig('Contour', bbox_inches='tight')