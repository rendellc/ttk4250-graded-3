# %% Imports
from scipy.io import loadmat
from scipy.stats import chi2

from sys import flags
interactive_mode = flags.interactive

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]


import numpy as np
from EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import ellipse
from vp_utils import detectTrees, odometry, Car
from utils import rotmat2d

import latexutils
import plotutils as plot


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

# %% Load data
VICTORIA_PARK_PATH = "./victoria_park/"
realSLAM_ws = {
    **loadmat(VICTORIA_PARK_PATH + "aa3_dr"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_lsr2"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_gpsx"),
}

timeOdo = (realSLAM_ws["time"] / 1000).ravel()
timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()   

steering = realSLAM_ws["steering"].ravel()
speed = realSLAM_ws["speed"].ravel()
LASER = (
    realSLAM_ws["LASER"] / 100
)  # Divide by 100 to be compatible with Python implementation of detectTrees
La_m = realSLAM_ws["La_m"].ravel()
Lo_m = realSLAM_ws["Lo_m"].ravel()

K = timeOdo.size
mK = timeLsr.size
Kgps = timeGps.size

# %% Parameters

L = 2.83  # axel distance
H = 0.76  # center to wheel encoder
a = 0.95  # laser distance in front of first axel
b = 0.5  # laser distance to the left of center

car = Car(L, H, a, b)

FIG_DIR = "real_results/"
parameters = dict(
    sigma_x = 0.1,
    sigma_y = 0.1,
    sigma_psi = np.deg2rad(1),
    sigma_range = 1,
    sigma_bearing = np.deg2rad(2.5),
    alpha_individual = 0.0001,
    alpha_joint = 0.0001,
    alpha_consistency = 0.05,
)
p = parameters


sigmas = [p["sigma_x"],p["sigma_y"],p["sigma_psi"]]
CorrCoeff = np.array([
    [1, 0, 0], 
    [0, 1, 0.9], 
    [0, 0.9, 1]])
Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)

R = np.diag([p["sigma_range"],p["sigma_bearing"]])**2

JCBBalphas = np.array([p["alpha_joint"], p["alpha_individual"]])

sensorOffset = np.array([car.a + car.L, car.b])
doAsso = True

slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas, sensor_offset=sensorOffset)

# For consistency testing
alpha = p["alpha_consistency"]
confprob = 1 - alpha

xupd = np.zeros((mK, 3))
a = [None] * mK
NIS = np.zeros(mK)
NISnorm = np.zeros(mK)
CI = np.zeros((mK, 2))
CI1 = np.array(chi2.interval(confprob, 1))
CI2 = np.array(chi2.interval(confprob, 2))
CInorm = np.zeros((mK, 2))
P_pose = np.zeros((K,3,3))
N_lmk = np.zeros(K)
GPSerror = np.zeros(mK)
NISxraw = np.zeros(mK)
NISyraw = np.zeros(mK)
NISxyraw = np.zeros(mK)
NISx = np.zeros(mK)
NISy = np.zeros(mK)
NISxy = np.zeros(mK)

# Initialize state
eta = np.array([Lo_m[0], La_m[0], 36 * np.pi / 180]) # you might want to tweak these for a good reference
P = np.zeros((3, 3))


mk_first = 1  # first seems to be a bit off in timing
mk = mk_first
t = timeOdo[0]

# %%  run
N = 10000

doPlot = True

lh_pose = None

# callback for clicking on plot
abort = False
def onclick(event):
    global abort
    if abs(event.x) + abs(event.y) < 50:
        abort = True 



if doPlot:
    fig, ax = plt.subplots(num=1, clear=True)
    fig.canvas.mpl_connect('button_press_event', onclick)

    lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
    sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
    sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

    figGps, axGps = plt.subplots()
    lineGpsError, = axGps.plot([],[],label="GPS error")
    axGps.legend()

do_raw_prediction = True
Rgps = np.diag([1,1])
if do_raw_prediction:  # TODO: further processing such as plotting
    Pc = P.copy()
    odos = np.zeros((K, 3))
    odox = np.zeros((K, 3))
    odox[0] = eta

    gpsk = 0
    for k in range(min(N, K - 1)):
        odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
        odox[k + 1], _ = slam.predict(odox[k], Pc, odos[k + 1])

        if gpsk < Kgps and k < K - 1 and timeGps[gpsk] <= timeOdo[k+1]:
            v = odox[k,:2] - np.array([Lo_m[gpsk], La_m[gpsk]])
            S = Pc[:2,:2] + Rgps
            NISxraw[gpsk] = v[0]**2 / S[0,0]
            NISyraw[gpsk] = v[1]**2 / S[1,1]
            NISxyraw[gpsk] = v.T @ np.linalg.solve(S, v)
            gpsk += 1

# %%

gpsk = 0
for k in tqdm(range(N)):
    P_pose[k] = P[:3,:3]
    N_lmk[k] = len(eta[3:])//2

    if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
        # Force P to symmetric: there are issues with long runs (>10000 steps)
        # seem like the prediction might be introducing some minor asymetries,
        # so best to force P symetric before update (where chol etc. is used).
        # TODO: remove this for short debug runs in order to see if there are small errors
        P = (P + P.T) / 2
        dt = timeLsr[mk] - t
        if dt < 0:  # avoid assertions as they can be optimized avay?
            raise ValueError("negative time increment")

        t = timeLsr[mk]  # ? reset time to this laser time for next post predict
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = slam.predict(eta, P, odo)

        z = detectTrees(LASER[mk])
        eta, P, NIS[mk], a[mk] = slam.update(eta, P, z)


        if timeGps[gpsk] <= timeOdo[k+1]:
            # Compute decompose NIS with GPS
            v = eta[:2] - np.array([Lo_m[gpsk], La_m[gpsk]])
            S = P[:2,:2] + Rgps
            NISx[gpsk] = v[0]**2 / S[0,0]
            NISy[gpsk] = v[1]**2 / S[1,1]
            NISxy[gpsk] = v.T @ np.linalg.solve(S, v)

            GPSerror[gpsk] = np.linalg.norm(v)

            lineGpsError.set_data(timeGps[:gpsk+1], GPSerror[:gpsk+1])
            axGps.relim()
            axGps.autoscale_view()

            if GPSerror[gpsk] > 25:
                print("gps error large, aborting")
                abort = True

            gpsk += 1
        

        num_asso = np.count_nonzero(a[mk] > -1)

        if num_asso > 0:
            NISnorm[mk] = NIS[mk] / (2 * num_asso)
            CInorm[mk] = np.array(chi2.interval(confprob, 2 * num_asso)) / (
                2 * num_asso
            )
        else:
            NISnorm[mk] = 1
            CInorm[mk].fill(1)

        xupd[mk] = eta[:3]

        if doPlot:
            sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
            if len(z) > 0:
                zinmap = (
                    rotmat2d(eta[2])
                    @ (
                        z[:, 0] * np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                        + slam.sensor_offset[:, None]
                    )
                    + eta[0:2, None]
                )
                sh_Z.set_offsets(zinmap.T)
            lh_pose.set_data(*xupd[mk_first:mk, :2].T)

            ax.set(
                xlim=[-200, 200],
                ylim=[-200, 200],
                title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
            )

            fig.canvas.draw()
            figGps.canvas.draw()

            #plt.draw()
            plt.pause(0.00001)

        mk += 1

    if k < K - 1:
        dt = timeOdo[k + 1] - t
        t = timeOdo[k + 1]
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = slam.predict(eta, P, odo)

    if abort:
        print("aborting")
        N = k
        break


# %% Consistency

# NIS
insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * (NISnorm[:mk] <= CInorm[:mk, 1])
insideCIxy = (CI2[0] <= NISxy[:gpsk]) * (NISxy[:gpsk] <= CI2[1])
insideCIx = (CI1[0] <= NISx[:gpsk]) * (NISx[:gpsk] <= CI1[1])
insideCIy = (CI1[0] <= NISy[:gpsk]) * (NISy[:gpsk] <= CI1[1])
insideCIxyraw = (CI2[0] <= NISxyraw[:gpsk]) * (NISxyraw[:gpsk] <= CI2[1])
insideCIxraw = (CI1[0] <= NISxraw[:gpsk]) * (NISxraw[:gpsk] <= CI1[1])
insideCIyraw = (CI1[0] <= NISyraw[:gpsk]) * (NISyraw[:gpsk] <= CI1[1])

fig3, ax3 = plt.subplots(num=3, clear=True)
nis_str = f"NIS ({insideCI.mean():.1%} inside)"
plot.pretty_NEESNIS(ax3, timeLsr[:mk], NISnorm[:mk], nis_str, CInorm[:mk,0], CInorm[:mk,1])
ax3.legend()
latexutils.save_fig(fig3, FIG_DIR + "NIS")


# %% slam

if do_raw_prediction:
    fig5, ax5 = plt.subplots(num=5, clear=True)
    ax5.scatter(
        Lo_m[timeGps < timeOdo[N - 1]],
        La_m[timeGps < timeOdo[N - 1]],
        c="r",
        marker=".",
        label="GPS",
    )
    ax5.plot(*odox[:N, :2].T, label="odom")
    ax5.grid()
    ax5.set_title("GPS vs odometry integration")
    ax5.legend()
    latexutils.save_fig(fig5, FIG_DIR + "GPSvsOdom")

    fig9, ax9 = plt.subplots(2, 1, num=9, clear=True)

    xy_str = f"NIS xy ({insideCIxyraw.mean():.1%} inside)"
    x_str = f"NIS x ({insideCIxraw.mean():.1%} inside)"
    y_str = f"NIS y ({insideCIyraw.mean():.1%} inside)"
    plot.pretty_NEESNIS(ax9[0], timeGps[:gpsk], NISxyraw[:gpsk], xy_str, CI2[0], CI2[1])
    plot.pretty_NEESNIS(ax9[1], timeGps[:gpsk], NISxraw[:gpsk], x_str, CI1[0], CI1[1])
    plot.pretty_NEESNIS(ax9[1], timeGps[:gpsk], NISyraw[:gpsk], y_str, CI1[0], CI1[1])

    for ax in ax9:
        #ax.set_xlim([timeOdo[0], timeOdo[N-1]])
        ax.legend()
    fig9.tight_layout()

    latexutils.save_fig(fig9, FIG_DIR + "NISraw")

    fig10, ax10s = plt.subplots(3, 1, sharex=True, num=10, clear=True)
    ax10s[0].plot(timeOdo[:N], odos[:N,0], label=r"odom $x$")
    ax10s[1].plot(timeOdo[:N], odos[:N,1], label=r"odom $y$")
    ax10s[2].plot(timeOdo[:N], odos[:N,2], label=r"odom $\psi$")
    ax10s[0].legend()
    ax10s[1].legend()
    ax10s[2].legend()

    latexutils.save_fig(fig10, FIG_DIR + "odom")
    


# %%
fig6, ax6 = plt.subplots(num=6, clear=True)
ax6.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x")
ax6.plot(*xupd[mk_first:mk, :2].T)
ax6.set(
    title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
)
latexutils.save_fig(fig6, FIG_DIR + "finalupdate")

# %%
fig7, ax7 = plt.subplots(num=7, clear=True)
P_norm = np.linalg.norm(P_pose[:mk], axis=(1,2))
ax7.plot(timeLsr[:mk], P_norm)
ax7.set_title("P pose norm")
latexutils.save_fig(fig7, FIG_DIR + "pose_covariance")

fig8, ax8 = plt.subplots(num=8, clear=True)
ax8.plot(timeOdo[:N], N_lmk[:N])
ax8.set_title("Number of landmarks over time")
latexutils.save_fig(fig8, FIG_DIR + "num_landmarks")

fig11, ax11 = plt.subplots(num=11, clear=True)
ax11.plot(timeLsr[:mk], np.rad2deg(xupd[:mk,2]))
ax11.set_title("Heading estimate (deg)")
latexutils.save_fig(fig11, FIG_DIR + "heading_estimate")


fig12, ax12 = plt.subplots(2, 1, num=12, clear=True)

xy_str = f"NIS xy ({insideCIxy.mean():.1%} inside)"
x_str = f"NIS x ({insideCIx.mean():.1%} inside)"
y_str = f"NIS y ({insideCIy.mean():.1%} inside)"
plot.pretty_NEESNIS(ax12[0], timeGps[:gpsk], NISxy[:gpsk], xy_str, CI2[0], CI2[1])
plot.pretty_NEESNIS(ax12[1], timeGps[:gpsk], NISx[:gpsk], x_str, CI1[0], CI1[1])
plot.pretty_NEESNIS(ax12[1], timeGps[:gpsk], NISy[:gpsk], y_str, CI1[0], CI1[1])

for ax in ax12:
    ax.legend()
fig12.tight_layout()

latexutils.save_fig(fig12, FIG_DIR + "NISxy")


if interactive_mode:
    plt.show(block=False)



# %%
