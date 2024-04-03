import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR

xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
num_points_x, num_points_y = 24, 24  # 20, 20  # You can adjust these numbers
x_values = np.linspace(xmin, xmax, num_points_x)
y_values = np.linspace(ymin, ymax, num_points_y)
x_mesh, y_mesh = np.meshgrid(x_values, y_values)
vec = jnp.column_stack((x_mesh.flatten(), y_mesh.flatten()))


################################
# DEFINE LOSS FUNCTIONS
################################
def norm(x):
    return (x / jnp.max(jnp.array([jnp.linalg.norm(x), 1e-8]))).squeeze()


def dot(x, y):
    # return jnp.clip(jnp.dot(x, y), -1.0, 1.0)
    return jnp.dot(x, y)


def cosine_distance(x, y):
    x, y = norm(x), norm(y)
    return (1 - dot(x, y)).squeeze()


def cos_similarity(x, y):
    x, y = norm(x), norm(y)
    return jnp.dot(x, y).squeeze()


def l2_loss(x, y):
    diff = jnp.subtract(x.squeeze(), y.squeeze())
    return jnp.sqrt(dot(diff, diff)).squeeze()


def l2n_loss(x, y):
    x, y = norm(x), norm(y)
    diff = jnp.subtract(x, y)
    return jnp.sqrt(dot(diff, diff)).squeeze()


def ang_distance(x, y):
    x, y = norm(x), norm(y)
    return jnp.arccos(dot(x, y)).squeeze()


def ang_distance_dp(x, y):
    x, y = norm(x), norm(y)
    d1 = ang_distance(x, y)
    d2 = ang_distance(-x, y)
    return jnp.min(jnp.array([d1, d2])).squeeze()


################################
# COMPUTE LOSS GRADIENTS
################################
distances = []
gradients = []
gradient_lengths = []
ground_truth = jnp.array([[1], [0]])
for distfunc in [cosine_distance, l2n_loss, l2_loss, ang_distance, ang_distance_dp]:
    distfunc_vmap = jax.vmap(distfunc, in_axes=(0, None))(vec, ground_truth)
    distfunc_grad = jax.grad(distfunc, argnums=0)
    distfunc_gradvmap = jax.vmap(distfunc_grad, in_axes=(0, None))(vec, ground_truth)
    distfunc_gradlength = jnp.linalg.norm(distfunc_gradvmap, axis=1)

    distances.append(distfunc_vmap)
    gradients.append(distfunc_gradvmap)
    gradient_lengths.append(distfunc_gradlength)


################################
# PLOT
################################
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

labels = [
    r"$d_{\mathrm{cd}}=1-\frac{y \cdot z}{\|y\|\|z\|}$",
    r"$d_{\mathrm{E,n}}=\sqrt{\left(\frac{y}{\|y\|}-\frac{z}{\|z\|}\right)\cdot \left(\frac{y}{\|y\|}-\frac{z}{\|z\|}\right)}$",
    r"$d_{\mathrm{E}}=\sqrt{(y-z)\cdot(y-z)}$",
    r"$d_{\mathrm{ang}}=\mathrm{arccos}\left( \frac{y \cdot z}{\|y\|\|z\|}\right)$",
    r"$\mathrm{min}\left(d_{\mathrm{ang}}(y,z), d_{\mathrm{ang}}(-y,z)\right)$",
]

scales = [5.0, 5.0, 5.0, 5.0]

fmin = float(jnp.min(jnp.array(gradient_lengths)))
fmax = float(jnp.max(jnp.array(gradient_lengths)))
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.2, clip=False)
cmap = matplotlib.colormaps["coolwarm_r"]  # rainbow

fig, ax = plt.subplots(1, len(gradients), sharey=True, figsize=(32, 6), gridspec_kw={"wspace": 0.1, "hspace": 0.1})

for i, axis in enumerate(ax):
    circle = plt.Circle((0, 0), 1, color="k", fill=False, linestyle="--", linewidth=2.0)
    axis.add_patch(circle)
    axis.set_aspect("equal")
    axis.set_xlim(-1.5, 1.5)
    axis.set_ylim(-1.5, 1.5)

    scaled_gradients = jnp.divide(gradients[i], jnp.linalg.norm(gradients[i], axis=1, keepdims=True))
    quiver_plot = axis.quiver(
        vec[:, 0],
        vec[:, 1],
        -1 * scaled_gradients[:, 0],
        -1 * scaled_gradients[:, 1],
        gradient_lengths[i],
        cmap=cmap,
        norm=norm,
        units="width",
        pivot="mid",
        scale=30.0,
        headwidth=3,
        width=0.01,
    )
    axis.plot(ground_truth[0], ground_truth[1], "o", color="k", markersize=10, alpha=0.7)
    axis.annotate(r"$z$", (ground_truth[0] + 0.05, ground_truth[1] + 0.1))
    axis.title.set_text(labels[i])
    axis.set_xlabel(r"$y_1$")

ax[0].set_ylabel(r"$y_2$")

cbar = fig.colorbar(quiver_plot, ax=ax, format="%.1f", ticks=[0, 0.5, 1.0], extend="max")
cbar.set_label(r"gradient length $\|\nabla_{y}d(y,z)\|$")
fig.suptitle(r"Negative gradients $-\nabla_{y}d(y,z)$ with z=[1,0]", fontsize=20)
out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", f"loss_gradients.pdf")
plt.savefig(out_p, bbox_inches="tight")
plt.show()
