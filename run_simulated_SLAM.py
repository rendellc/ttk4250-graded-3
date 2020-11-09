# %% Imports
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import chi2
import utils

import latexutils
import plotutils as plot

from sys import flags
interactive_mode = flags.interactive


try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm to have progress bar")

    # def tqdm as dummy as it is not available
    def tqdm(*args, **kwargs):
        return args[0]

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )



from EKFSLAM import EKFSLAM
from plotting import ellipse

latexutils.set_save_dir("sim_results")
parameters = dict(
    sigma_x = 0.025,
    sigma_y = 0.02,
    sigma_psi = np.deg2rad(0.37),
    sigma_range = 0.06,
    sigma_bearing = np.deg2rad(1.2),
    alpha_individual = 1e-5,
    alpha_joint = 1e-5,
    alpha_consistency = 0.05,
)
p = parameters
doAsso = True
doAssoPlot = False
playMovie = False
saveMovie = False

latexutils.save_params_to_csv(
    latexutils.parameter_to_texvalues(parameters),
    "parameters.csv")


# %% Load data
simSLAM_ws = loadmat("simulatedSLAM")

z = [zk.T for zk in simSLAM_ws["z"].ravel()]
landmarks = simSLAM_ws["landmarks"].T
odometry = simSLAM_ws["odometry"].T
poseGT = simSLAM_ws["poseGT"].T



K = len(z) 
M = len(landmarks)

# %% Initilize
Q = np.diag([p["sigma_x"],p["sigma_y"],p["sigma_psi"]])**2
R = np.diag([p["sigma_range"],p["sigma_bearing"]])**2


JCBBalphas = np.array([p["alpha_joint"], p["alpha_individual"]])
slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas)

# allocate
eta_pred: List[Optional[np.ndarray]] = [None] * K
P_pred: List[Optional[np.ndarray]] = [None] * K
eta_hat: List[Optional[np.ndarray]] = [None] * K
P_hat: List[Optional[np.ndarray]] = [None] * K
a: List[Optional[np.ndarray]] = [None] * K
NIS = np.zeros(K)
NIS_range = np.zeros(K)
NIS_bearing = np.zeros(K)
NISnorm_range = np.zeros(K)
NISnorm_bearing = np.zeros(K)
NISnorm = np.zeros(K)
CI = np.zeros((K, 2))
CInorm = np.zeros((K, 2))
CI_rangebearing = np.zeros((K, 2))
CInorm_rangebearing = np.zeros((K, 2))
NEESes = np.zeros((K, 3))

# For consistency testing
alpha = p["alpha_consistency"]
confprob = 1 - alpha

# init
eta_pred[0] = poseGT[0]  # we start at the correct position for reference
P_pred[0] = np.zeros((3, 3))  # we also say that we are 100% sure about that

# %% Set up plotting
# plotting

if doAssoPlot:
    figAsso, axAsso = plt.subplots( clear=True)

# %% Run simulation
N = K
t = np.arange(N)

print("starting sim (" + str(N) + " iterations)")

for k, z_k in tqdm(enumerate(z[:N])):

    eta_hat[k], P_hat[k], NIS[k], a[k], NIS_range[k], NIS_bearing[k] = slam.update(eta_pred[k], P_pred[k], z_k)

    if k < K - 1:
        eta_pred[k + 1], P_pred[k + 1] = slam.predict(eta_hat[k], P_hat[k], odometry[k])

    assert (
        eta_hat[k].shape[0] == P_hat[k].shape[0]
    ), "dimensions of mean and covariance do not match"

    num_asso = np.count_nonzero(a[k] > -1)

    CI_rangebearing[k] = chi2.interval(confprob, num_asso)
    CI[k] = chi2.interval(confprob, 2 * num_asso)

    if num_asso > 0:
        NISnorm_range[k] = NIS_range[k] / num_asso
        NISnorm_bearing[k] = NIS_bearing[k] / num_asso
        NISnorm[k] = NIS[k] / (2 * num_asso)
        CInorm[k] = CI[k] / (2 * num_asso)
        CInorm_rangebearing[k] = CI_rangebearing[k] / num_asso
    else:
        NISnorm_range[k] = 1
        NISnorm_bearing[k] = 1
        NISnorm[k] = 1
        CInorm_rangebearing[k].fill(1)
        CInorm[k].fill(1)

    NEESes[k] = slam.NEESes(eta_hat[k][:3], P_hat[k][:3,:3], poseGT[k])

    if doAssoPlot and k > 0:
        axAsso.clear()
        axAsso.grid()
        zpred = slam.h(eta_pred[k]).reshape(-1, 2)
        axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
        axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
        xcoords = np.block([[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
        ycoords = np.block([[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
        for x, y in zip(xcoords, ycoords):
            axAsso.plot(x, y, lw=3, c="r")
        axAsso.legend()
        axAsso.set_title(f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
        plt.draw()
        plt.pause(0.001)


print("sim complete")

pose_est = np.array([x[:3] for x in eta_hat[:N]])
lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat[:N]]
lmk_est_final = lmk_est[N - 1]

np.set_printoptions(precision=4, linewidth=100)

# %% Plotting of results
mins = np.amin(landmarks, axis=0)
maxs = np.amax(landmarks, axis=0)

ranges = maxs - mins
offsets = ranges * 0.2

mins -= offsets
maxs += offsets

fig2, ax2 = plt.subplots(clear=True)
# landmarks
ax2.scatter(*landmarks.T, c="r", marker="^")
ax2.scatter(*lmk_est_final.T, c="b", marker=".")
# Draw covariance ellipsis of measurements
for l, lmk_l in enumerate(lmk_est_final):
    idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
    rI = P_hat[N - 1][idxs, idxs]
    el = ellipse(lmk_l, rI, 5, 200)
    ax2.plot(*el.T, "b")

ax2.plot(*poseGT.T[:2], c="r", label="gt")
ax2.plot(*pose_est.T[:2], c="g", label="est")
ax2.plot(*ellipse(pose_est[-1, :2], P_hat[N - 1][:2, :2], 5, 200).T, c="g")
ax2.set(title="results", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
ax2.axis("equal")
ax2.grid()

latexutils.save_fig(fig2, "trajectory.pdf")

# %% Consistency
CI1 = chi2.interval(confprob, 1)
CI2 = chi2.interval(confprob, 2)
CI3 = chi2.interval(confprob, 3)
CI1N = np.array(chi2.interval(confprob, 1*N)) / N
CI2N = np.array(chi2.interval(confprob, 2*N)) / N
CI3N = np.array(chi2.interval(confprob, 3*N)) / N
df_anis = 2 * sum([np.count_nonzero(ak > -1) for ak in a])
CIANIS = np.array(chi2.interval(confprob, df_anis))/len(NIS)
df_rangebearing = sum([np.count_nonzero(ak > -1) for ak in a])
CIANIS_rangebearing = np.array(chi2.interval(confprob, df_rangebearing))/len(NIS_bearing)

NEESpose, NEESpos, NEESpsi = NEESes.T
insideCIpose = (CI3[0] <= NEESpose) * (NEESpose <= CI3[1])
insideCIpos = (CI2[0] <= NEESpos) * (NEESpos <= CI2[1])
insideCIpsi = (CI1[0] <= NEESpsi) * (NEESpsi <= CI1[1])
insideCIrange = (CInorm_rangebearing[:N,0] <= NISnorm_range[:N]) * (NISnorm_range[:N] <= CInorm_rangebearing[:N,1])
insideCIbearing = (CInorm_rangebearing[:N,0] <= NISnorm_bearing[:N]) * (NISnorm_bearing[:N] <= CInorm_rangebearing[:N,1])


ANEESpose = NEESpose.mean()
ANEESpos = NEESpos.mean()
ANEESpsi = NEESpsi.mean()
ANIS = NIS.mean()
ANIS_range = NIS_range.mean()
ANIS_bearing = NIS_bearing.mean()

insideCI = (CInorm[:N,0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N,1])

consistencydatas = [
        dict(avg=ANEESpose,inside=insideCIpose.mean(), text="NEES pose",CI=CI3N),
        dict(avg=ANEESpos,inside=insideCIpos.mean(), text="NEES pos",CI=CI2N),
        dict(avg=ANEESpsi,inside=insideCIpsi.mean(), text="NEES psi",CI=CI1N),
        dict(avg=ANIS,inside=insideCI.mean(), text="NIS",CI=CIANIS),
        dict(avg=ANIS_range,inside=insideCIrange.mean(), text="NIS range",CI=CIANIS_rangebearing),
        dict(avg=ANIS_bearing,inside=insideCIbearing.mean(), text="NIS bearing",CI=CIANIS_rangebearing),
]

latexutils.save_consistency_results(consistencydatas, "consistency.csv")


print(f"{'ANEESpose':<20} {ANEESpose:<20}\t{CI3N}")
print(f"{'ANEESpos':<20} {ANEESpos:<20}\t{CI2N}")
print(f"{'ANEESpsi':<20} {ANEESpsi:<20}\t{CI1N}")
print(f"{'ANIS':<20} {ANIS:<20.3f}\t{CIANIS}")
print(f"{'ANIS range':<20} {ANIS_range:<20.3f}\t{CIANIS_rangebearing}")
print(f"{'ANIS bearing':<20} {ANIS_bearing:<20.3f}\t{CIANIS_rangebearing}")
print(f"{'NIS':<20} {insideCI.mean():.1%} inside")

# NIS

fig3, ax3 = plt.subplots(clear=True)
nis_str = f"NIS ({insideCI.mean():.1%} inside)"
plot.pretty_NEESNIS(ax3, t, NISnorm[:N], nis_str, CInorm[:N,0], CInorm[:N,1])
ax3.legend(loc="upper right")
#ax3.set_title(f'NIS, {insideCI.mean():.0%} inside CI')

latexutils.save_fig(fig3, "NIS.pdf")

# NEES

fig4, ax4 = plt.subplots(nrows=3, ncols=1, clear=True, sharex=True)
pose_str = f"NEES pose ({insideCIpose.mean():.1%} inside)"
plot.pretty_NEESNIS(ax4[0], t, NEESpose, pose_str, CI3[0], CI3[1])
pos_str = f"NEES pos ({insideCIpos.mean():.1%} inside)"
plot.pretty_NEESNIS(ax4[1], t, NEESpos, pos_str, CI2[0], CI2[1])
psi_str = f"NEES heading ({insideCIpsi.mean():.1%} inside)"
plot.pretty_NEESNIS(ax4[2], t, NEESpsi, psi_str, CI1[0], CI1[1])

for ax in ax4:
    ax.legend(loc="upper right")

tags = ['all', 'pos', 'heading']
dfs = [3, 2, 1]

fig4.tight_layout()

latexutils.save_fig(fig4, "NEES.pdf")


# Decomposed NISes
fig, axs = plt.subplots(2,1, sharex=True)
range_str = f"NIS range ({insideCIrange.mean():.1%} inside)"
plot.pretty_NEESNIS(axs[0], t, NISnorm_range[:N], range_str, CInorm_rangebearing[:N,0], CInorm_rangebearing[:N,1])
bearing_str = f"NIS bearing ({insideCIbearing.mean():.1%} inside)"
plot.pretty_NEESNIS(axs[1], t, NISnorm_bearing[:N], bearing_str, CInorm_rangebearing[:N,0], CInorm_rangebearing[:N,1])
for ax in axs:
    ax.legend(loc="upper right")
fig.tight_layout()

latexutils.save_fig(fig, "NIS_decomposed.pdf")


# %% RMSE
pos_err = np.linalg.norm(pose_est[:N,:2] - poseGT[:N,:2], axis=1)
heading_err = np.rad2deg(np.abs(utils.wrapToPi(pose_est[:N,2] - poseGT[:N,2])))
pos_rmse = np.sqrt((pos_err**2).mean())
heading_rmse = np.sqrt((heading_err**2).mean())

fig5, ax5 = plt.subplots(nrows=2, ncols=1, clear=True, sharex=True)

ax5[0].plot(t,pos_err, label=f"Position error ({pos_rmse:.3f})")
#ax5[0].plot([t[0],t[~0]], [pos_rmse]*2, label="Position RMSE")
ax5[0].set_ylabel("[m]")
#ax5[0].set_title(f"pos: RMSE {} [m]")
ax5[1].plot(t,heading_err, label=f"Heading error ({heading_rmse:.3f})")
#ax5[1].plot([t[0],t[~0]], [heading_rmse]*2, label="Heading RMSE")
ax5[1].set_ylabel("[deg]")
#ax5[1].set_title(f"heading: RMSE {np.sqrt((heading_err**2).mean()):.4f} [deg]")

for ax in ax5:
    ax.set_xlim(t[0], t[~0])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.grid(True)

fig5.tight_layout()
fig5.align_ylabels()

latexutils.save_fig(fig5, "RMSE.pdf")

# %% Movie time

if playMovie:
    try:
        print("recording movie...")

        from celluloid import Camera

        pauseTime = 0.05
        fig_movie, ax_movie = plt.subplots(clear=True)

        camera = Camera(fig_movie)

        ax_movie.grid()
        ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
        camera.snap()

        for k in tqdm(range(N)):
            ax_movie.scatter(*landmarks.T, c="r", marker="^")
            ax_movie.plot(*poseGT[:k, :2].T, "r-")
            ax_movie.plot(*pose_est[:k, :2].T, "g-")
            ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

            if k > 0:
                el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 5, 200)
                ax_movie.plot(*el.T, "g")

            numLmk = lmk_est[k].shape[0]
            for l, lmk_l in enumerate(lmk_est[k]):
                idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
                rI = P_hat[k][idxs, idxs]
                el = ellipse(lmk_l, rI, 5, 200)
                ax_movie.plot(*el.T, "b")

            camera.snap()
        animation = camera.animate(interval=100, blit=True, repeat=False)
        if saveMovie:
            animation.save(
                    latexutils.SAVE_DIR / "animation.mp4",
                    dpi=400,
                    savefig_kwargs={
                        "pad_inches": "tight",
                    },
            )


    except ImportError:
        print(
            "Install celluloid module, \n\n$ pip install celluloid\n\nto get fancy animation of EKFSLAM."
        )


if interactive_mode:
    plt.show(block=False)


# %%
