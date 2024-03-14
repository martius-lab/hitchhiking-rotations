import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from pathlib import Path
import pandas as pd
from hitchhiking_rotations.utils import RotRep


files = [str(s) for s in Path(os.path.join(HITCHHIKING_ROOT_DIR, "results", "pose_to_cube_image")).rglob("*result.npy")]
results = [np.load(file, allow_pickle=True) for file in files]

# trainer_name
methods_res = {}
selected_metric = "l2"
rename_and_filter = True

df_res = {}
df_res["method"] = []
df_res["score"] = []

for run in results:
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

        methods_res[k].append(v)


df = pd.DataFrame.from_dict(df_res)

if rename_and_filter:
    mapping = {
        "r9": RotRep.ROTMAT,
        "r6": RotRep.RSIX,
        "quat_c": RotRep.QUAT_C,
        "quat_rf": str(RotRep.QUAT) + "_rf",
        "rotvec": RotRep.EXP,
        "euler": RotRep.EULER,
    }

    training_metric = "l2"
    for k, v in mapping.items():
        df["method"][df["method"] == k + "_" + training_metric] = v

    df["method"] = pd.Categorical(df["method"], categories=[v for v in mapping.values()], ordered=True)

plt.style.use(os.path.join(HITCHHIKING_ROOT_DIR, "assets", "prettyplots.mplstyle"))
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})
plt.figure(figsize=(7, 2.5))
plt.subplot(1, 1, 1)


sns.boxplot(data=df, x="score", y="method", palette="Greens", orient="h", width=0.5, linewidth=1.5, fliersize=2.5)
plt.xlabel("Error - MSE")
plt.ylabel("")
plt.tight_layout()

out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", "pose_to_cube_image", "figure_12b.pdf")
plt.savefig(out_p)
plt.show()
