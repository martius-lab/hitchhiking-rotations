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

rot = Rotation.random(1)  # generate N random rotations
rotmat = jnp.array(rot.as_matrix()).reshape((3,3))
predmats = jax.random.uniform(key=jax.random.PRNGKey(np.random.randint(0,10000)), shape=(3, 3), minval=-2.0, maxval=2.0)

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
    return norm(rotmat, gso(predmat_vec.reshape(3, 2)))

def norm_svd(predmat_vec, rotmat):
    return norm(rotmat, svd(predmat_vec.reshape(3, 3)))

grads_gso = jax.grad(norm_gso)
grads_svd = jax.grad(norm_svd)

epochs = 300
stepsize = 1.

predmats_gso, predmats_svd = predmats.reshape(9,)[:6], predmats.reshape(9,)
predmats_gso_list, predmats_svd_list = [], []
rotmat_gso, rotmat_svd = [], []
gradgso, gradsvd = jnp.zeros((6,)), jnp.zeros((9,))

def momentum(grad, grad_prev, beta=0.9):
    return beta * grad_prev + (1-beta) * grad

for i in range(epochs):
    predmats_gso_list.append(predmats_gso)
    gradgso = momentum(grads_gso(predmats_gso, rotmat), gradgso)
    predmats_gso -= stepsize * gradgso
    rotmat_gso.append(gso(predmats_gso.reshape(3, 2)))

    predmats_svd_list.append(predmats_svd)
    gradsvd = momentum(grads_svd(predmats_svd, rotmat), gradsvd)
    predmats_svd -= stepsize * gradsvd
    rotmat_svd.append(svd(predmats_svd.reshape(3, 3)))


def plot_matrices(ax, mat_list, labels, frame):
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    colors = ["r", "g", "b", "y", "m", "c"]

    offsetvec = jnp.zeros((3,))
    for i in range(len(mat_list)):
        mat = mat_list[i]

        for j in range(len(mat)):
            if j == 0:
                ax.quiver(
                    offsetvec, offsetvec, offsetvec,
                    mat[0][j], mat[1][j], mat[2][j],
                    color=colors[i], label=labels[i]
                )
            else:
                ax.quiver(offsetvec, offsetvec, offsetvec,
                          mat[0][j], mat[1][j], mat[2][j], color=colors[i])

            ax.text(mat[0][j], mat[1][j], mat[2][j], f"$e_{j + 1}$", color="black")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Step {frame}")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
#ax.set_axis_off()

def update_plot(frame):
    ax.clear()
    #ax.set_axis_off()
    plot_matrices(ax, [rotmat, rotmat_gso[frame], rotmat_svd[frame]], ["RotMat", "GSO", "SVD"], frame)

# Create animation
ani = animation.FuncAnimation(fig, update_plot, frames=len(rotmat_gso), interval=10)

plt.show()

save_path = os.path.join(os.getcwd(), "gso_vs_svd.gif")
ani.save(save_path, writer="ffmpeg", fps=60)