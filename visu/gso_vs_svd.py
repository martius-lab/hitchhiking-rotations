import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib.colors as matcolors
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


# from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
# from mpl_toolkits.mplot3d import Axes3D

# import lovely_jax as lj
# lj.monkey_patch()

N = int(1e3)  # Number of randomly sampled representation vectors
plot_frames = False  # Plot frames before/after SVD/GSO transform
plot_frames_with_grads = False
plot_grads = False  # Plot ratio between gradients entries
plot_ratios = True  # Plot 2D scatter plot of gradients
plot_condnums = False  # Plot condition numbers of Hessian matrices and max eigen values


# Helper functions to rearrange 6D vectors to 3x2 matrices
# where columns denote the representation vectors
def mat2vec(mat: jnp.ndarray, dimb=3) -> jnp.ndarray:
    # Same as table.T.reshape(-1, 1)
    return rearrange(mat, "a b -> (b a)", a=3, b=dimb)


def vec2mat(vec: jnp.ndarray, dimb=3) -> jnp.ndarray:
    return rearrange(vec, "(b a) -> a b", a=3, b=dimb)


def bmat2bvec(mat: jnp.ndarray, dimb=3) -> jnp.ndarray:
    return rearrange(mat, "i a b -> i (b a)", a=3, b=dimb)


def bvec2bmat(vec: jnp.ndarray, dimb=3) -> jnp.ndarray:
    return rearrange(vec, "i (b a) -> i a b", a=3, b=dimb)


# rot = Rotation.random(N)  # generate N random rotations
# rotmats = jnp.array(rot.as_matrix())
rotmats = jnp.tile(jnp.eye(3), (N, 1, 1))
rotmats_vec = bmat2bvec(rotmats)

predmats = rotmats + 1e-1 * jax.random.normal(key=jax.random.PRNGKey(42), shape=(N, 3, 3))
# predmats = jax.random.uniform(key=jax.random.PRNGKey(42), shape=(N, 3, 3), minval=-2.0, maxval=2.0)
# predmats = 2*jax.random.normal(key=jax.random.PRNGKey(1), shape=(N, 3, 3))
predmats_vec = bmat2bvec(predmats)

assert (
    (
        (
            (
                jnp.allclose(predmats[0, :, 0], predmats_vec[0, :3])
                and jnp.allclose(predmats[0, :, 1], predmats_vec[0, 3:6])
            )
            and jnp.allclose(predmats[0, 2, 1], predmats_vec[0, 5])
        )
        and jnp.allclose(predmats[0, :, :], bvec2bmat(predmats_vec)[0])
    )
    and jnp.allclose(predmats[0, :, :], vec2mat(predmats_vec[0]))
) and jnp.allclose(mat2vec(predmats[0, :, :]), predmats_vec[0]), "Conversion functions are not working"


@jax.jit
def gso(m: jnp.ndarray) -> jnp.ndarray:
    """Gram-Schmidt orthogonalization from 6D input.
    Source: Google research - https://github.com/google-research/google-research/blob/193eb9d7b643ee5064cb37fd8e6e3ecde78737dc/special_orthogonalization/utils.py#L93-L115
    """
    x = m[:, 0]
    y = m[:, 1]
    xn = x / jnp.linalg.norm(x, axis=0)
    z = jnp.cross(xn, y)
    zn = z / jnp.linalg.norm(z, axis=0)
    yn = jnp.cross(zn, xn)
    return jnp.c_[xn, yn, zn]


@jax.jit
def svd(m: jnp.ndarray) -> jnp.ndarray:
    """Maps 3x3 matrices onto SO(3) via symmetric orthogonalization.
    Source: Google research - https://github.com/google-research/google-research/blob/193eb9d7b643ee5064cb37fd8e6e3ecde78737dc/special_orthogonalization/utils.py#L93-L115
    """
    """
    m = jax.lax.cond(jnp.linalg.matrix_rank(m) < 3,
                     true_fun=lambda x: x + jnp.eye(3) * 1e-10,
                     false_fun=lambda x: x,
                     operand=m)
    """
    U, _, Vh = jnp.linalg.svd(m, full_matrices=False)
    det = jnp.linalg.det(jnp.matmul(U, Vh))
    return jnp.matmul(jnp.c_[U[:, :-1], U[:, -1] * det], Vh)


gso_vmap = jax.vmap(gso)
pred_gso = gso_vmap(predmats)
rot_gso = gso_vmap(rotmats)

svd_vmap = jax.vmap(svd)
pred_svd = svd_vmap(predmats)
rot_svd = svd_vmap(rotmats)


def plot_matrix(
    ax,
    mat,
    color,
    label,
    offset=jnp.zeros(
        3,
    ),
):
    for i in range(len(mat)):
        if i == 0:
            ax.quiver(
                offset[0][i], offset[1][i], offset[2][i], mat[0][i], mat[1][i], mat[2][i], color=color, label=f"{label}"
            )
        else:
            ax.quiver(offset[0][i], offset[1][i], offset[2][i], mat[0][i], mat[1][i], mat[2][i], color=color)
        ax.text(mat[0][i], mat[1][i], mat[2][i], f"$e_{i + 1}$", color="black")


def plot_matrices(ax, mat_list, labels, off_list=None):
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    colors = ["r", "g", "b", "y", "m", "c"]

    if off_list is None:
        off_list = [jnp.zeros((3, 3)) for _ in range(len(mat_list))]

    for i in range(len(mat_list)):
        plot_matrix(ax, mat_list[i], colors[i], label=labels[i], offset=off_list[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()


# PLOT frames to check that GSO and SVD layers are working
if plot_frames:
    for r00, r01, r02, r10, r11, r12 in zip(rotmats, rot_svd, rot_gso, predmats, pred_svd, pred_gso):
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection="3d", proj_type="ortho")
        plot_matrices(ax1, [r00, r01, r02], ["rotmat", "rot_svd", "rot_gso"])

        ax2 = fig.add_subplot(122, projection="3d", proj_type="ortho")
        plot_matrices(ax2, [r10, r11, r12], ["predmat", "pred_svd", "pred_gso"])

        plt.show()


###############################################################################
# DEFINE GRADIENTS AND HESSIANS
###############################################################################


def norm(mat1: jnp.ndarray, mat2: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(mat1.flatten() - mat2.flatten())


def norm_gso(predmat_vec: jnp.ndarray, rotmat: jnp.ndarray) -> jnp.ndarray:
    return norm(rotmat, gso(vec2mat(predmat_vec, dimb=2)))


def norm_svd(predmat_vec: jnp.ndarray, rotmat: jnp.ndarray) -> jnp.ndarray:
    return norm(rotmat, svd(vec2mat(predmat_vec)))


def hess_gso(rotmat: jnp.ndarray, predmat_vec: jnp.ndarray) -> jnp.ndarray:
    return jax.hessian(norm_gso)(predmat_vec, rotmat)


def hess_svd(rotmat: jnp.ndarray, predmat_vec: jnp.ndarray) -> jnp.ndarray:
    return jax.hessian(norm_svd)(predmat_vec, rotmat)


###############################################################################
# COMPUTE GRADIENTS
###############################################################################

loss_svd = jax.vmap(norm, (0, 0))(rotmats, pred_svd)
loss_gso = jax.vmap(norm, (0, 0))(rotmats, pred_gso)

grads_gso = jax.vmap(jax.grad(norm_gso), (0, 0))(predmats_vec[:, :6], rotmats)
grads_svd = jax.vmap(jax.grad(norm_svd), (0, 0))(predmats_vec, rotmats)
gradnorm_gso = jnp.linalg.norm(grads_gso, axis=-1)
gradnorm_svd = jnp.linalg.norm(grads_svd, axis=-1)

gradnorm1_gso = jnp.linalg.norm(grads_gso[:, 0:3], axis=-1)
gradnorm2_gso = jnp.linalg.norm(grads_gso[:, 3:6], axis=-1)

gradnorm1_svd = jnp.linalg.norm(grads_svd[:, 0:3], axis=-1)
gradnorm2_svd = jnp.linalg.norm(grads_svd[:, 3:6], axis=-1)
gradnorm3_svd = jnp.linalg.norm(grads_svd[:, 6:9], axis=-1)

ratios12_gso = jnp.divide(gradnorm1_gso, gradnorm2_gso)

ratios12_svd = jnp.divide(gradnorm1_svd, gradnorm2_svd)
ratios13_svd = jnp.divide(gradnorm1_svd, gradnorm3_svd)
ratios23_svd = jnp.divide(gradnorm2_svd, gradnorm3_svd)

if plot_frames_with_grads:
    for r0, r_svd, r_gso, g_svd, g_gso in zip(rotmats, pred_svd, pred_gso, grads_svd, grads_gso):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
        g_gso = jnp.c_[
            vec2mat(g_gso, dimb=2),
            np.zeros(
                3,
            ),
        ]
        r_list = [r0, r_svd, r_gso, -1 * vec2mat(g_svd), -1 * g_gso]
        off_list = [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3))] + [r_svd, r_gso]

        plot_matrices(ax, r_list, ["rotmat", "svd", "gso", "grad_svd", "grad_gso"], off_list)
        plt.show()

###############################################################################
# COMPUTE HESSIANS
###############################################################################

hessmats_gso = jax.vmap(hess_gso, (0, 0))(rotmats, predmats_vec[:, :6])
hessmats_svd = jax.vmap(hess_svd, (0, 0))(rotmats, predmats_vec)

eig_gso = jnp.sort(jax.vmap(jnp.linalg.eig)(hessmats_gso)[0], axis=-1)
eig_svd = jnp.sort(jax.vmap(jnp.linalg.eig)(hessmats_svd)[0], axis=-1)
condnums_gso = jnp.divide(jnp.abs(eig_gso[:, -1]), jnp.abs(eig_gso[:, 0]))
condnums_svd = jnp.divide(jnp.abs(eig_svd[:, -1]), jnp.abs(eig_svd[:, 0]))


###############################################################################
# ANALYSE GRADIENTS & HESSIANS
###############################################################################

df = pd.DataFrame(
    {
        "loss": np.r_[loss_svd, loss_gso].flatten(),
        "gradnorm": np.r_[gradnorm_svd, gradnorm_gso].flatten(),
        "ratios12": np.r_[ratios12_svd, ratios12_gso].flatten(),
        "ratios13": np.r_[ratios13_svd, [None] * rot_gso.shape[0]].flatten(),
        "ratios23": np.r_[ratios23_svd, [None] * rot_gso.shape[0]].flatten(),
        "condnums": np.r_[condnums_svd, condnums_gso].flatten(),
        "eigmin": np.r_[eig_svd[:, 0], eig_gso[:, 0]].flatten(),
        "eigmax": np.r_[eig_svd[:, -1], eig_gso[:, -1]].flatten(),
        # Note: eig_svd[:, -1] is the max eigenvalue as the eigenvalues are sorted
    }
)

df["Label"] = ["SVD"] * rot_svd.shape[0] + ["GSO"] * rot_gso.shape[0]


def boxplot_labels(labels, labeltext=None):
    fig, axs = plt.subplots(1, len(labels), sharey=True)
    for i in range(len(labels)):
        # sns.histplot(data=df, x=labels[i], hue="Label", bins=50, ax=axs[i])
        sns.boxplot(x="Label", y=labels[i], data=df, ax=axs[i], showfliers=True, palette="Blues")
        # sns.violinplot(x="Label", y=labels[i], data=df, ax=axs[i])
        # sns.stripplot(x="Label", y=labels[i], data=df, ax=axs[i], color="black", alpha=0.01)
        axs[i].set_yscale("log")
        axs[i].tick_params(labelsize=18)

        if labeltext is not None:
            axs[i].set_xlabel(labeltext[i], fontsize=18)
        else:
            axs[i].set_xlabel(labels[i], fontsize=18)

    axs[0].set_ylabel("Count", fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_2D(labelx, labely, legend=None, plottype="kde"):
    assert len(labelx) == len(labely) == len(legend), "labelx, labely and legend must have the same length"
    n = len(labelx)
    fig, axs = plt.subplots(2, n, sharex=True, sharey=True)

    labels = ["SVD", "GSO"]
    if plottype == "scatter" and legend is not None:
        for i in range(2):
            for j in range(n):
                idx = i * n + j
                if idx < 4:
                    points = axs[i, j].scatter(
                        df[labelx[j]][df["Label"] == labels[i]],
                        df[labely[j]][df["Label"] == labels[i]],
                        c=df[legend[j]][df["Label"] == labels[i]],
                        s=20,
                        cmap="Spectral_r",
                        norm=matcolors.LogNorm(),
                    )  # set style options
                    axs[i, j].set_xscale("log")
                    axs[i, j].set_yscale("log")
                    axs[i, j].set_xlabel(labelx[j])
                    axs[i, j].set_ylabel(labely[j])
                    axs[i, j].set_title(f"{labels[i]}")
                    plt.colorbar(points, label=legend[j])

    elif plottype == "kde":
        for i in range(2):
            for j in range(n):
                idx = i * n + j
                if idx < 4:
                    sns.kdeplot(
                        x=df[labelx[j]][df["Label"] == labels[i]],
                        y=df[labely[j]][df["Label"] == labels[i]],
                        # norm=matcolors.LogNorm(),
                        ax=axs[i, j],
                        cmap="Spectral_r",  # cmap="Reds",'Greens',#
                        fill=True,
                        levels=30,
                        log_scale=(False, True),
                        cbar=True,
                        clip=((None, None), (None, None)),
                    )
                    axs[i, j].set_xlabel(labelx[j])
                    axs[i, j].set_ylabel(labely[j])
                    axs[i, j].set_title(f"{labels[i]}")
                    axs[i, j].grid(axis="y", linestyle="--")

                else:
                    axs[i, j].axis("off")

    plt.show()


def plot_2D_paper():
    labels = ["GSO", "SVD", "SVD", "SVD"]
    dataname = ["ratios12", "ratios12", "ratios13", "ratios23"]
    datalabels = [
        r"$\|\nabla_{\nu_1}L\| / \|\nabla_{\nu_2}L\|$",
        r"$\|\nabla_{m_1}L\| / \|\nabla_{m_2}L\|$",
        r"$\|\nabla_{m_1}L\| / \|\nabla_{m_3}L\|$",
        r"$\|\nabla_{m_2}L\| / \|\nabla_{m_3}L\|$",
    ]

    n = len(labels)

    fig, axs = plt.subplots(1, n, sharey=True)
    axs = axs.ravel()

    for i in range(n):
        cnt = sns.kdeplot(
            x=df["loss"][df["Label"] == labels[i]],
            y=df[dataname[i]][df["Label"] == labels[i]],
            # norm=matcolors.LogNorm(),
            ax=axs[i],
            cmap="coolwarm",  #'Greens',#"Spectral_r", #cmap="Reds",
            fill=True,
            levels=50,
            log_scale=(False, True),
            # cbar=True,
            clip=((None, None), (None, None)),
            antialiased=True,
            # thresh=0.1,
        )
        for c in cnt.collections:
            c.set_edgecolor("face")

        axs[i].set_ylim(0.06, 120)
        axs[i].set_xlabel(r"Loss $L(R,r)$")
        axs[i].set_ylabel(None)
        axs[i].set_title(datalabels[i])
        plt.text(0.02, 0.98, labels[i], ha="left", va="top", fontsize=14, color="black", transform=axs[i].transAxes)
        axs[i].grid(axis="y", linestyle="--")
        if i == 0:
            axs[i].set_facecolor("gray")

    plt.show()


if plot_ratios:
    plot_2D_paper()
    # plot_2D(["loss", "loss", "loss"],
    #        ["ratios12", "ratios13", "ratios23"],
    #        ["condnums", "condnums", "condnums"],
    #        plottype="kde")

if plot_grads:
    boxplot_labels(["ratios12", "ratios13", "ratios23"])

if plot_condnums:
    # \mathrm{abs}(\lambda_{\mathrm{max}}) \mathrm{abs}(\lambda_{\mathrm{min}}})
    labeltext = [
        r"Condition number $\mathrm{abs}(\lambda_{\mathrm{max}}) / \mathrm{abs}(\lambda_{\mathrm{min}}) $",
        r"Maximal eigenvalue $\lambda_{\mathrm{max}}$",
    ]
    boxplot_labels(["condnums", "eigmax"], labeltext=labeltext)
