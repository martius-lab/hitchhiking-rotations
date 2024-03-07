import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from pathlib import Path
import pandas as pd
from hitchhiking_rotations.utils import RotRep

nb_max = 6
nb_values = range(1, nb_max + 1)
files = [
    str(s)
    for nb in nb_values
    for s in Path(os.path.join(HITCHHIKING_ROOT_DIR, "results", f"pose_to_fourier_{nb}")).rglob("*result.npy")
]

# files = [str(s) for s in Path(os.path.join(HITCHHIKING_ROOT_DIR, "results", "pose_to_fourier_1")).rglob("*result.npy")]
results = [np.load(file, allow_pickle=True) for file in files]

# trainer_name
methods_res = {}
selected_metric = "l2"
rename_and_filter = True

df_res = {}
df_res["method"] = []
df_res["score"] = []
df_res["basis"] = []
df_res["func"] = []

for ib, run in enumerate(results):
    for trainer_name, logging_dict in run.items():
        if trainer_name.find("test") == -1:
            continue

        # only use metrics generated for testing
        metrics_test = logging_dict["test"]
        k = trainer_name[:-5]

        if k not in methods_res:
            # create a list for each seed
            methods_res[k] = []

        v = metrics_test[selected_metric]["sum"] / metrics_test[selected_metric]["count"]
        df_res["method"].append(k)
        df_res["score"].append(v)

        # File path example: "/hitchhiking-rotations/results/pose_to_fourier_1/seed_2_result.npy"
        df_res["basis"].append(int(files[ib].split("/")[-2].split("_")[-1]))
        df_res["func"].append(int(files[ib].split("/")[-1].split("_")[-2]))

        methods_res[k].append(v)


df = pd.DataFrame.from_dict(df_res)

if rename_and_filter:
    mapping = {
        "r9": RotRep.SVD,
        "r6": RotRep.GSO,
        "quat_c": RotRep.QUAT_C,
        "quat_rf": str(RotRep.QUAT) + "_rf",
        "rotvec": RotRep.EXP,
        "euler": RotRep.EULER,
    }

    training_metric = "l2"
    df["method"] = df["method"].replace({k + "_" + training_metric: v for k, v in mapping.items()})
    df["method"] = pd.Categorical(df["method"], categories=[v for v in mapping.values()], ordered=True)

plt.style.use(os.path.join(HITCHHIKING_ROOT_DIR, "assets", "prettyplots.mplstyle"))
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})

plt.figure(figsize=(5.5, 1.0))
g = sns.catplot(
    data=df,
    x="basis",
    y="score",
    hue="method",
    kind="box",
    palette="Greens",
    flierprops={"markeredgecolor": "grey"},
    height=7.0,
    aspect=2.0,
)

sns.move_legend(g, "upper left", bbox_to_anchor=(0.11, 0.98), ncol=3, title="Network input")  # len(names)

for i in range(nb_max - 1):
    plt.axvline(0.5 + i, color="lightgrey", dashes=(2, 2))

plt.xlabel(f"Error - {selected_metric}")
plt.ylabel("")
plt.yscale("log")
plt.tight_layout()

out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", "figure_14.pdf")
plt.savefig(out_p)
plt.show()
