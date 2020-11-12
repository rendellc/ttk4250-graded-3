import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BIGPLOTSIZE = (3.5, 6)
HEATMAP_COLORS = [[0,1,0,1],[1,0,0,1]]
HEATMAP_CM = LinearSegmentedColormap.from_list("heatmap_cm",HEATMAP_COLORS)

def trajectoryPlot3D(x_est, x_true, z_GNSS, N, GNSSk):
    # 3d position plot
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1, projection='3d')

    ax.plot3D(x_est[:N, 1], x_est[:N, 0], -x_est[:N, 2])
    if len(x_true) > 0:
        ax.plot3D(x_true[:N, 1], x_true[:N, 0], -x_true[:N, 2])

    ax.plot3D(z_GNSS[:GNSSk, 1], z_GNSS[:GNSSk, 0], -z_GNSS[:GNSSk, 2])
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")

    return fig, ax


def kinematicplot(t, x_nom, eul, N, title):
    fig, axs = plt.subplots(3, 1, num=11, sharex=True, clear=True)
    axs[0].plot(t, x_nom[:N, POS_IDX])
    axs[0].set(title="Position [m]")
    axs[0].legend(["N", "E", "D"])
    axs[1].plot(t, x_nom[:N, VEL_IDX])
    axs[1].set(title="Velocity [m/s]")
    axs[1].legend(["N", "E", "D"])
    axs[2].plot(t, np.rad2deg(eul[:N]))
    axs[2].set(title="Euler angles [deg]")
    axs[2].legend([r"$\phi$", r"$\theta$", r"$\psi$"])
    fig.tight_layout()

    return fig

def biasplot(t, x_nom, eul, N, title):
    fig, axs = plt.subplots(2, 1, num=12, sharex=True, clear=True)
    axs[0].plot(t, x_nom[:N, ACC_BIAS_IDX])
    axs[0].set(title="Acc [m/s^2]")
    axs[0].legend(["$x$", "$y$", "$z$"])
    axs[1].plot(t, np.rad2deg(x_nom[:N, GYRO_BIAS_IDX]) * 3600)
    axs[1].set(title="Gyro [deg/h]")
    axs[1].legend(["$x$", "$y$", "$z$"])
    #fig.suptitle(title)
    fig.tight_layout()

    return fig


def stateplot(t, x_nom, eul, N, title):
    figall, axs = plt.subplots(5, 1, num=2, sharex=True, clear=True)
    figpos, axpos = plt.subplots(1, 1, clear=True)
    figvel, axvel = plt.subplots(1, 1, clear=True)
    figang, axang = plt.subplots(1, 1, clear=True)
    figab, axab = plt.subplots(1, 1, clear=True)
    figgb, axgb = plt.subplots(1, 1, clear=True)

    _multiplot([axs[0], axpos], t, x_nom[:N, POS_IDX], r"Pos $[m]$",
            ["N", "E", "D"], "upper right"
    )

    _multiplot([axs[1], axvel], t, x_nom[:N, VEL_IDX], r"Vel $[m]$",
            ["N", "E", "D"], "upper right"
    )

    _multiplot([axs[2], axang], t, np.rad2deg(eul[:N]), "Attitude [deg]",
            [r"$\phi$", r"$\theta$", r"$\psi$"], 'upper right'
    )

    _multiplot([axs[3], axab], t, x_nom[:N, ACC_BIAS_IDX], r"Acl bias $[m/s^2]$",
            ["$x$", "$y$", "$z$"], 'upper right'
    )

    _multiplot([axs[4], axgb], t, np.rad2deg(x_nom[:N, GYRO_BIAS_IDX]) * 3600, "Gyro bias [deg/h]",
            ["$x$", "$y$", "$z$"], 'upper right'
    )
    axs[4].set_ylim(GYRO_BIAS_YLIM)
    axgb.set_ylim(GYRO_BIAS_YLIM)

    #figall.suptitle(title)
    figall.set_size_inches(BIGPLOTSIZE)
    figall.tight_layout()
    figall.align_ylabels()


    #return fig
    return figall, figpos, figvel, figang, figab, figgb

def kinematicerrorplot(t, delta_x, eul_error, N, title):
    fig, axs = plt.subplots(3, 1, clear=True, sharex=True)
    delta_x_RMSE = np.round(np.sqrt(np.mean(delta_x[:N] ** 2, axis=0)),3)
    axs[0].plot(t, delta_x[:N, POS_IDX])
    axs[0].set(title="position error [m]")
    axs[0].legend(
        [
            #f"N ({delta_x_RMSE[POS_IDX[0]]})",
            #f"E ({delta_x_RMSE[POS_IDX[1]]})",
            #f"D ({delta_x_RMSE[POS_IDX[2]]})",
            f"N",
            f"E",
            f"D",
        ],
        loc='upper right'
    )

    axs[1].plot(t, delta_x[:N, VEL_IDX])
    axs[1].set(title="Velocities error [m]")
    axs[1].legend(
        [
            # f"N ({delta_x_RMSE[VEL_IDX[0]]})",
            # f"E ({delta_x_RMSE[VEL_IDX[1]]})",
            # f"D ({delta_x_RMSE[VEL_IDX[2]]})",
            f"N",
            f"E",
            f"D",
        ],
        loc='upper right'
    )

    axs[2].plot(t, eul_error)
    axs[2].set(title="Euler angles error [deg]")
    axs[2].legend(
        [
            #rf"$\phi$ ({np.sqrt(np.mean((eul_error[:N, 0])**2)):.3f})",
            #rf"$\theta$ ({np.sqrt(np.mean((eul_error[:N, 1])**2)):.3f})",
            #rf"$\psi$ ({np.sqrt(np.mean((eul_error[:N, 2])**2)):.3f})",
            r"$\phi$",
            r"$\theta$",
            r"$\psi$",
        ],
        loc='best'
    )

    fig.tight_layout()
    #fig.suptitle(title)

    return fig

def biaserrorplot(t, delta_x, eul_error, N, title):
    fig, axs = plt.subplots(2, 1, clear=True, sharex=True)
    delta_x_RMSE = np.round(np.sqrt(np.mean(delta_x[:N] ** 2, axis=0)),3)

    axs[0].plot(t, delta_x[:N, ERR_ACC_BIAS_IDX])
    axs[0].set(title="Accl bias error [m/s^2]")
    axs[0].legend(
        [
            f"$x$ ({delta_x_RMSE[ERR_ACC_BIAS_IDX[0]]})",
            f"$y$ ({delta_x_RMSE[ERR_ACC_BIAS_IDX[1]]})",
            f"$z$ ({delta_x_RMSE[ERR_ACC_BIAS_IDX[2]]})",
        ],
        loc='upper right'
    )

    axs[1].plot(t, np.rad2deg(delta_x[:N, ERR_GYRO_BIAS_IDX]))
    axs[1].set(title="Gyro bias error [deg/s]")
    axs[1].legend(
        [
            f"$x$ ({np.rad2deg(delta_x_RMSE[ERR_GYRO_BIAS_IDX[0]]):.3f})",
            f"$y$ ({np.rad2deg(delta_x_RMSE[ERR_GYRO_BIAS_IDX[1]]):.3f})",
            f"$z$ ({np.rad2deg(delta_x_RMSE[ERR_GYRO_BIAS_IDX[2]]):.3f})",
        ],
        loc='upper right'
    )

    fig.tight_layout()
    #fig.suptitle(title)

    return fig


def _multiplot(axs, x, y, ylabel, labels, loc):
    """
    make the same plot on several axes
    """
    for ax in axs:
        ax.plot(x, y)
        ax.set(ylabel=ylabel)
        ax.legend(labels, loc=loc)

def stateerrorplot(t, delta_x, eul_error, N, title):
    figall, axs = plt.subplots(5, 1, clear=True, sharex=True)

    figpos, axpos = plt.subplots(1, 1, clear=True)
    figvel, axvel = plt.subplots(1, 1, clear=True)
    figang, axang = plt.subplots(1, 1, clear=True)
    figab, axab = plt.subplots(1, 1, clear=True)
    figgb, axgb = plt.subplots(1, 1, clear=True)

    delta_x_RMSE = np.sqrt(np.mean(delta_x[:N] ** 2, axis=0))
    _multiplot([axs[0],axpos], t, delta_x[:N, POS_IDX], r"Pos $[m]$", [
            f"N ({delta_x_RMSE[POS_IDX[0]]:.3f})",
            f"E ({delta_x_RMSE[POS_IDX[1]]:.3f})",
            f"D ({delta_x_RMSE[POS_IDX[2]]:.3f})",
        ], 'upper right'
    )

    _multiplot([axs[1],axvel], t, delta_x[:N, VEL_IDX], r"Vel $[m]$", [
            f"N ({delta_x_RMSE[VEL_IDX[0]]:.3f})",
            f"E ({delta_x_RMSE[VEL_IDX[1]]:.3f})",
            f"D ({delta_x_RMSE[VEL_IDX[2]]:.3f})",
        ], 'upper right'
    )

    _multiplot([axs[2],axang], t, eul_error, "Attitude [deg]", [
            rf"$\phi$ ({np.sqrt(np.mean((eul_error[:N, 0])**2)):.3f})",
            rf"$\theta$ ({np.sqrt(np.mean((eul_error[:N, 1])**2)):.3f})",
            rf"$\psi$ ({np.sqrt(np.mean((eul_error[:N, 2])**2)):3f})",
        ], 'upper right'
    )

    _multiplot([axs[3],axab], t, delta_x[:N, ERR_ACC_BIAS_IDX], r"Acl bias $[m/s^2]$", [
            f"$x$ ({delta_x_RMSE[ERR_ACC_BIAS_IDX[0]]:.3f})",
            f"$y$ ({delta_x_RMSE[ERR_ACC_BIAS_IDX[1]]:.3f})",
            f"$z$ ({delta_x_RMSE[ERR_ACC_BIAS_IDX[2]]:.3f})",
        ], 'upper right'
    )

    _multiplot([axs[4],axgb], t, np.rad2deg(delta_x[:N, ERR_GYRO_BIAS_IDX]), "Gyro bias [deg/s]", [
            f"$x$ ({np.rad2deg(delta_x_RMSE[ERR_GYRO_BIAS_IDX[0]]):.2e})",
            f"$y$ ({np.rad2deg(delta_x_RMSE[ERR_GYRO_BIAS_IDX[1]]):.2e})",
            f"$z$ ({np.rad2deg(delta_x_RMSE[ERR_GYRO_BIAS_IDX[2]]):.2e})",
        ], 'upper right'
    )

    #fig.suptitle(title)
    figall.set_size_inches(BIGPLOTSIZE)
    figall.tight_layout()
    figall.align_ylabels()

    #figall.set_size_inches(4

    return figall, figpos, figvel, figang, figab, figgb

def boxplot(ax, datas, ndim, labels):
    N = len(datas[0])
    gauss_compare = np.sum(np.random.randn(ndim, N)**2, axis=0)
    ax.boxplot([*datas, gauss_compare], notch=True, showfliers=False)
    ax.set_xticklabels([*labels, 'gauss'], rotation=90)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="minor", length=0)

def pretty_NEESNIS(ax, t, neesnis, label, CIlow, CIhigh, fillCI=False, color="tab:gray",lineCI=True):
    ax.plot(t, neesnis, label=label)

    CIhighmax = np.max(1.5*CIhigh) # y limit if decided by CI
    yUpper = np.max([CIhighmax, *neesnis])
    ax.fill_between(t, np.zeros_like(CIlow), CIlow, facecolor=color, alpha=0.5) 
    ax.fill_between(t, CIhigh, yUpper, facecolor=color, alpha=0.5)

    if lineCI:
        ax.plot(t, np.ones_like(t)*CIlow, c=color, linewidth=0.5)
        ax.plot(t, np.ones_like(t)*CIhigh, c=color, linewidth=0.5)

    ax.set_xlim([t[0], t[~0]])

    ax.set_ylim([0, yUpper])


def plot_NEES(ax, delta_x, P, t, state_indices):
    NEES = [eskf._NEES(P[k][state_indices**2], delta_x[k][state_indices]) for k in range(len(t))]
    ax.plot(t,NEES)


def plot_NIS(NIS, CI, NIS_name, confprob, dt, N, GNSSk, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(NIS[:GNSSk])
    ax.plot(np.array([0, N - 1]) * dt, (CI @ np.ones((1, 2))).T)
    insideCI = np.mean((CI[0] <= NIS[:GNSSk]) * (NIS[:GNSSk] <= CI[1]))
    
    upperY = min(np.max(NIS), CI[1]//1 * 4)
    ax.set_title(f"{NIS_name} ({100 *  insideCI:.1f} % inside {100 * confprob} CI)")
    ax.set_xlim(0,(N-1)*dt)
    ax.set_ylim(0,upperY)

def heatmap(fig, ax, pos, weights, xlim=None, ylim=None, bins=50, cmin=None, wmin=None, wmax=None):
    assert len(pos) == len(weights)

    if xlim is None:
        xlim = [np.min(pos[:,0]), np.max(pos[:,0])]
    if ylim is None:
        ylim = [np.min(pos[:,1]), np.max(pos[:,1])]
    if wmin is None:
        wmin = np.min(weights)
    if wmax is None:
        wmax = np.max(weights)

    pos_normalized = np.vstack([
        (pos[:,0] - xlim[0])/(xlim[1] - xlim[0]),
        (pos[:,1] - ylim[0])/(ylim[1] - ylim[0])
    ]).T

    assert pos.shape == pos_normalized.shape

    region = np.zeros((bins,bins))
    counter = np.zeros_like(region)
    pos_indices = np.asarray(bins*pos_normalized, dtype=int)

    for k, (xi,yi) in enumerate(pos_indices):
        if 0 <= xi < bins and 0 <= yi < bins:
            region[yi,xi] += weights[k]
            counter[yi,xi] += 1

    # compute average
    mask = counter != 0
    region[mask] = region[mask]/counter[mask]
    np.clip(region, wmin, wmax, out=region)
    # normalize from 0 to 1 for propper colormapping
    region = (region - wmin)/(wmax-wmin) 

    image = HEATMAP_CM(region)
    image[counter == 0, 3] = 0.0
    #alpha = counter.copy()
    #alpha[mask] = alpha[mask]/np.max(counter)


    #alpha[mask] = 1.0 # set all alphas to 1 where a count has occured




    im = ax.imshow(image, 
            extent=(*xlim, *ylim), interpolation="bilinear",cmap=HEATMAP_CM,
            aspect="equal", origin="lower")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    colorbar = fig.colorbar(im, ax=ax)
    colorbar.ax.locator_params(nbins=3)
    colorbar.ax.set_yticklabels([f"{wmin:.3f}", '', f">{wmax:.3f}"])

    #ax.hist2d(pos[:,0], pos[:,1], weights=weights, bins=bins, range=[xlim, ylim], cmin=cmin)
    return colorbar


def splitplot(axtop, axbottom, xdata, ydata, label, y_skips, topscale=1):
    axbottom.plot(xdata, ydata, label=label)
    axtop.plot(xdata, ydata, label=label)

    axbottom.set_ylim(np.min(ydata), y_skips[0])
    axtop.set_ylim(y_skips[1], 150)
    axbottom.set_xlim(xdata[0], xdata[~0])

    axtop.spines["bottom"].set_visible(False)
    axtop.xaxis.tick_top()
    axtop.tick_params(labeltop=False)

    axbottom.spines["top"].set_visible(False)
    axbottom.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    dtop = d/topscale
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axtop.transAxes, color='k', clip_on=False)
    axtop.plot((-d, +d), (-dtop, +dtop), **kwargs)        # top-left diagonal
    axtop.plot((1 - d, 1 + d), (-dtop, +dtop), **kwargs)  # top-right diagonal

    kwargs.update(transform=axbottom.transAxes)  # switch to the bottom axes
    axbottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axbottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

