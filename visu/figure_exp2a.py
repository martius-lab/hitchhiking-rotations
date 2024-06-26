import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from pathlib import Path
import pandas as pd
from hitchhiking_rotations.utils import RotRep


selected_metric = "geodesic_distance"
training_metric = "chordal_distance"
exps = ["pcd_to_pose", "cube_image_to_pose"]

assert len(sys.argv) == 2, "Please provide the experiment name as an argument."
assert sys.argv[1] in exps, f"Experiment name should be one of {exps}"

exp = sys.argv[1]

files = [str(s) for s in Path(os.path.join(HITCHHIKING_ROOT_DIR, "results", exp)).rglob("*result.npy")]
results = [np.load(file, allow_pickle=True) for file in files]

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

        v = metrics_test[selected_metric]["sum"] / metrics_test[selected_metric]["count"]

        df_res["method"].append(k)
        df_res["score"].append(v)


df = pd.DataFrame.from_dict(df_res)

mapping = {
    "r9_svd": RotRep.SVD,
    "r6_gso": RotRep.GSO,
    "quat_c": RotRep.QUAT_C,
    # "quat_rf": RotRep.QUAT_RF,
    "rotvec": RotRep.EXP,
    "euler": RotRep.EULER,
}

df["method"] = df["method"].replace({k + "_" + training_metric: v for k, v in mapping.items()})
df["method"] = pd.Categorical(df["method"], categories=[v for v in mapping.values()], ordered=True)

plt.style.use(os.path.join(HITCHHIKING_ROOT_DIR, "assets", "prettyplots.mplstyle"))
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})
plt.figure(figsize=(5, 2.5))
plt.subplot(1, 1, 1)


sns.boxplot(
    data=df,
    x="score",
    y="method",
    palette="Blues",
    orient="h",
    width=0.5,
    linewidth=1.5,
    fliersize=2.5,
    showfliers=True,
)

if selected_metric == "geodesic_distance":
    plt.xlabel("Geodesic distance")

plt.ylabel("")
# plt.xscale("log")

print("WARNING: Tick labels are hardcoded!")
plt.xticks([0.3, 0.4, 0.5, 0.6], ["0.3", "0.4", "0.5", "0.6"])

plt.tight_layout()
out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", exp, "figure_exp2a.pdf")

plt.savefig(out_p)
plt.show()
