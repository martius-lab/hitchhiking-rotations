import os
import numpy as np
from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as matcolors
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from einops import rearrange

epochs = 300
stepsize = 1.0


# Helper functions to rearrange 6D vectors to 3x2 matrices
# where columns denote the representation vectors
def mat2vec(mat, dimb=3):
    # Same as table.T.reshape(-1, 1)
    return rearrange(mat, "a b -> (b a)", a=3, b=dimb)


def vec2mat(vec, dimb=3):
    return rearrange(vec, "(b a) -> a b", a=3, b=dimb)


# rot = Rotation.random(1)  # generate N random rotations
# rotmat = jnp.array(rot.as_matrix()).reshape((3,3))
rotmat = jnp.eye(3)
predmats = jax.random.uniform(
    key=jax.random.PRNGKey(np.random.randint(0, 10000)), shape=(3, 3), minval=-2.0, maxval=2.0
)

if False:
    # Set representation vectors close to -e_3
    predmats = predmats.at[:, 0].set(-1 * rotmat[:, 2] + 0.01)
    predmats = predmats.at[:, 1].set(-1 * rotmat[:, 2] + 0.02)
    predmats = predmats.at[:, 2].set(-1 * rotmat[:, 2])

if False:
    # Set representation vectors close to -e_2
    predmats = predmats.at[:, 0].set(-1 * rotmat[:, 0] + 0.01)
    predmats = predmats.at[:, 1].set(-1 * rotmat[:, 0] + 0.02)
    predmats = predmats.at[:, 2].set(-1 * rotmat[:, 0])

if False:
    # Left handed coordinate system breaks SVD
    predmats = predmats.at[:, 0].set(-1 * rotmat[:, 0])
    predmats = predmats.at[:, 1].set(-1 * rotmat[:, 1])
    predmats = predmats.at[:, 2].set(-1 * rotmat[:, 2])

if False:
    # Large ratios between feature vectors
    predmats = predmats.at[:, 0].set(-3 * rotmat[:, 0] + 0.01)
    predmats = predmats.at[:, 1].set(-0.001 * rotmat[:, 1])
    predmats = predmats.at[:, 2].set(1 * rotmat[:, 2])

if False:
    predmats = predmats.at[:, 1].set(predmats[:, 0])


@jax.jit
def gso(m: jnp.ndarray) -> jnp.ndarray:
    """Gram-Schmidt orthogonalization from 6D input.
    Source: Google research - https://github.com/google-research/google-research/blob/193eb9d7b643ee5064cb37fd8e6e3ecde78737dc/special_orthogonalization/utils.py#L93-L115
    """
    x, y = m[:, 0], m[:, 1]
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
    U, _, Vh = jnp.linalg.svd(m, full_matrices=False)
    det = jnp.linalg.det(jnp.matmul(U, Vh))
    return jnp.matmul(jnp.c_[U[:, :-1], U[:, -1] * det], Vh)


def norm(mat1: jnp.ndarray, mat2: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(mat1.flatten() - mat2.flatten())


def norm_gso(predmat_vec, rotmat):
    predmat_mat = vec2mat(predmat_vec, dimb=2)
    return norm(rotmat, gso(predmat_mat))


def norm_svd(predmat_vec, rotmat):
    predmat_mat = vec2mat(predmat_vec)
    return norm(rotmat, svd(predmat_mat))


grads_gso = jax.grad(norm_gso)
grads_svd = jax.grad(norm_svd)

predmats_gso, predmats_svd = mat2vec(predmats)[:6], mat2vec(predmats)
predmats_gso_list, predmats_svd_list = [], []
predmats_gso_list.append(predmats_gso)
predmats_svd_list.append(predmats_svd)
rotmat_gso, rotmat_svd = [], []
gradgso, gradsvd = jnp.zeros((6,)), jnp.zeros((9,))


def momentum(grad_current, grad_prev, beta=0.9):
    return beta * grad_prev + (1 - beta) * grad_current


for i in range(epochs):
    gradgso = momentum(grads_gso(predmats_gso, rotmat), gradgso)
    predmats_gso -= stepsize * gradgso
    tmp1 = rearrange(predmats_gso, "(b a) -> a b", a=3, b=2)
    rotmat_gso.append(gso(tmp1))
    predmats_gso_list.append(predmats_gso)

    gradsvd = momentum(grads_svd(predmats_svd, rotmat), gradsvd)
    predmats_svd -= stepsize * gradsvd
    tmp2 = rearrange(predmats_svd, "(b a) -> a b", a=3, b=3)
    rotmat_svd.append(svd(tmp2))
    predmats_svd_list.append(predmats_svd)

predmats_gso_list = jnp.array(predmats_gso_list)
predmats_svd_list = jnp.array(predmats_svd_list)

lims = 4.0


def plot_matrices(ax, mat_list, labels, frame):
    ax.set(xlim=(-0.5 * lims, lims), ylim=(-0.5 * lims, lims), zlim=(-0.5 * lims, lims))
    colors = ["r", "g", "b", "y", "m", "c"]

    offsetvec = jnp.zeros((3,))
    for i in range(len(mat_list)):
        mat = mat_list[i]

        for j in range(len(mat)):
            if j == 0:
                ax.quiver(
                    offsetvec, offsetvec, offsetvec, mat[0][j], mat[1][j], mat[2][j], color=colors[i], label=labels[i]
                )
            else:
                ax.quiver(offsetvec, offsetvec, offsetvec, mat[0][j], mat[1][j], mat[2][j], color=colors[i])

            ax.text(mat[0][j], mat[1][j], mat[2][j], f"$e_{j + 1}$", color="black")


fig = plt.figure(figsize=(10, 10), facecolor="w")
ax = fig.add_subplot(111, projection="3d", facecolor="w")
# ax.set_axis_off()
ax.view_init(elev=5, azim=30)
ax.dist = 3
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove margins


def update_plot(frame):
    ax.clear()
    # ax.set_axis_off()
    plot_matrices(ax, [rotmat, rotmat_gso[frame], rotmat_svd[frame]], ["RotMat", "GSO", "SVD"], frame)

    def plot_line(data, index, color="r", linestyle="-"):
        ax.plot(data[:, index], data[:, index + 1], data[:, index + 2], lw=2, color=color, linestyle=linestyle)

    plot_line(predmats_gso_list[: frame + 1], index=0, color="darkgreen")
    plot_line(predmats_gso_list[: frame + 1], index=3, color="lightgreen")
    plot_line(predmats_svd_list[: frame + 1], index=0, color="darkblue")
    plot_line(predmats_svd_list[: frame + 1], index=3, color="blue")
    plot_line(predmats_svd_list[: frame + 1], index=6, color="lightblue")

    ax.plot(
        [predmats_gso_list[frame, 0], rotmat_gso[frame][0][0]],
        [predmats_gso_list[frame, 1], rotmat_gso[frame][1][0]],
        [predmats_gso_list[frame, 2], rotmat_gso[frame][2][0]],
        color="grey",
        linestyle=":",
    )
    ax.plot(
        [predmats_gso_list[frame, 3], rotmat_gso[frame][0][1]],
        [predmats_gso_list[frame, 4], rotmat_gso[frame][1][1]],
        [predmats_gso_list[frame, 5], rotmat_gso[frame][2][1]],
        color="grey",
        linestyle=":",
    )

    ax.plot(
        [predmats_svd_list[frame, 0], rotmat_svd[frame][0][0]],
        [predmats_svd_list[frame, 1], rotmat_svd[frame][1][0]],
        [predmats_svd_list[frame, 2], rotmat_svd[frame][2][0]],
        color="grey",
        linestyle=":",
    )
    ax.plot(
        [predmats_svd_list[frame, 3], rotmat_svd[frame][0][1]],
        [predmats_svd_list[frame, 4], rotmat_svd[frame][1][1]],
        [predmats_svd_list[frame, 5], rotmat_svd[frame][2][1]],
        color="grey",
        linestyle=":",
    )
    ax.plot(
        [predmats_svd_list[frame, 6], rotmat_svd[frame][0][2]],
        [predmats_svd_list[frame, 7], rotmat_svd[frame][1][2]],
        [predmats_svd_list[frame, 8], rotmat_svd[frame][2][2]],
        color="grey",
        linestyle=":",
    )

    ax.plot([0, lims], [0, 0], [0, 0], color="black", linestyle="--")
    ax.plot([0, 0], [0, lims], [0, 0], color="black", linestyle="--")
    ax.plot([0, 0], [0, 0], [0, lims], color="black", linestyle="--")

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_zaxis().set_ticklabels([])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Step {frame}")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()


# Create animation
ani = animation.FuncAnimation(fig, update_plot, frames=len(rotmat_gso), interval=10)

plt.show()

save_path = os.path.join(os.getcwd(), "gso_vs_svd.gif")
ani.save(save_path, writer="ffmpeg", fps=10)
