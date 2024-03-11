import os
import numpy as np
from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR

# SETTINGS
# plot_type = 'Paper'
plot_type = "Appendix"
N = int(4e6)  # int(2e5)  # number of random rotations
N_pairs = int(4e5)  # int(2e4

# USE SCIPY TO COMPUTE DISTANCES
# Approach: Randomly sample SO(3) matrices then compute representation vectors
my_rot = Rotation.random(N)  # generate N random rotations
my_rot = my_rot[np.random.choice(N, 2 * N_pairs, replace=False)]  # generate N_pairs pairs of rotations
rot1, rot2 = my_rot[N_pairs:], my_rot[:N_pairs]
dist_so3 = np.linalg.norm(rot1.as_matrix() - rot2.as_matrix(), axis=(1, 2), ord="fro")
l = rot1.as_matrix().shape[0]


def eucl_norm(mat1, mat2):
    return jnp.linalg.norm(mat1 - mat2)


# Rotation representations
mat1 = jnp.array(rot1.as_matrix().reshape(l, -1))
mat2 = jnp.array(rot2.as_matrix().reshape(l, -1))
euler1 = jnp.array(rot1.as_euler("xyz", degrees=False))
euler2 = jnp.array(rot2.as_euler("xyz", degrees=False))
quat1 = rot1.as_quat(canonical=False)
quat2 = rot2.as_quat(canonical=False)
exp1 = jnp.array(rot1.as_rotvec())
exp2 = jnp.array(rot2.as_rotvec())
mrp1 = jnp.array(rot1.as_mrp())
mrp2 = jnp.array(rot2.as_mrp())

# Distances
dist_so3_L2 = jax.vmap(eucl_norm)(mat1, mat2)
dist_euler = jax.vmap(eucl_norm)(euler1, euler2)
dist_quat = jax.vmap(eucl_norm)(quat1, quat2)
dist_exp = jax.vmap(eucl_norm)(exp1, exp2)
dist_mrp = jax.vmap(eucl_norm)(mrp1, mrp2)

size = 9
linewidth = 3.0
max_so3 = np.max(dist_so3)
s2 = 2 * np.sqrt(2)
max_exp_coord = 2 * np.pi
max_mrp = 2
max_euler_angle = 2 * np.pi

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams.update({"font.size": 18})

colors = [
    (0.368, 0.507, 0.71),
    (0.881, 0.611, 0.142),
    (0.923, 0.386, 0.209),
    (0.56, 0.692, 0.195),
    (0.528, 0.471, 0.701),
    (0.772, 0.432, 0.102),
    (0.572, 0.586, 0.0),
]

if plot_type == "Paper":
    dist_list = [dist_so3_L2, dist_euler, dist_quat]
    limit_list = [None, max_euler_angle, 2]
    text_list = [
        "Rotation\nmatrix\nas vector\nin " + r"$\mathbb{R}^9$",
        "Euler\nangles",
        "Quaternions",
        "Modified Rodrigues\nparameters",
    ]
    ticklist = [[0, max_so3], [0, 2 * np.pi], [0, 2]]
    ticklabels = [[r"$0$", r"$2\sqrt{2}$"], [r"$0$", r"$2\pi$"], [r"$0$", r"$2$"]]
    slope = [1, np.pi / s2, np.sqrt(2) / s2]
    linecolors = [colors[2], colors[2], colors[2]]
    markercolors = [colors[0], colors[0], colors[0]]

    fig, axs = plt.subplots(ncols=len(dist_list), nrows=1, tight_layout=True)
    fig.set_figheight(5)
    fig.set_figwidth(9)

if plot_type == "Appendix":
    dist_list = [dist_so3_L2, dist_euler, dist_exp, dist_quat, dist_mrp]
    limit_list = [None, max_euler_angle, max_exp_coord, 2, max_mrp]
    text_list = [
        "Rotation matrix\nas vector in " + r"$\mathbb{R}^9$",
        "Euler\nangles",
        "Exponential\ncoordinates",
        "Quaternions",
        "Modified Rodrigues\nparameters",
    ]
    ticklist = [[0, max_so3], [0, 2 * np.pi], [0, 2 * np.pi], [0, 2], [0, 2]]
    ticklabels = [
        [r"$0$", r"$2\sqrt{2}$"],
        [r"$0$", r"$2\pi$"],
        [r"$0$", r"$2\pi$"],
        [r"$0$", r"$2$"],
        [r"$0$", r"$2$"],
    ]
    slope = [1, np.pi / s2, np.pi / s2, np.sqrt(2) / s2, np.tan(np.pi / 4) / s2]
    linecolors = [colors[2], colors[2], colors[2], colors[2], colors[2]]
    markercolors = [colors[0], colors[0], colors[0], colors[0], colors[0]]

    fig, axs = plt.subplots(ncols=len(dist_list), nrows=1, tight_layout=True)
    fig.set_figheight(6)
    fig.set_figwidth(18)

for i in range(len(dist_list)):
    axs[i].margins(x=0)
    axs[i].plot(dist_so3, dist_list[i], ".", alpha=0.01, markersize=size, color=markercolors[i])
    axs[i].plot([0, max_so3], [0, max_so3 * slope[i]], "k--", linewidth=linewidth)
    if limit_list[i] is not None:
        axs[i].plot(
            [0, max_so3], [limit_list[i], limit_list[i]], "--", color=linecolors[i], alpha=0.8, linewidth=linewidth
        )

    if plot_type == "Paper" and i == 0:
        axs[i].text(0.05, 0.68, text_list[i], transform=axs[i].transAxes)
    else:
        axs[i].text(0.05, 0.76, text_list[i], transform=axs[i].transAxes)

    axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axs[i].set_yticks(ticklist[i])
    axs[i].set_yticklabels(ticklabels[i])
    axs[i].set_xlabel(r"$\|R_1 - R_2\|_{\mathrm{F}}$")

axs[0].set_ylabel(r"$\|r_1-r_2\|_2$")
plt.subplots_adjust(wspace=0.0, hspace=0.0)
out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", f"lipschitz_constants_{plot_type}.png")
plt.savefig(out_p, bbox_inches="tight", dpi=200)
plt.show()
