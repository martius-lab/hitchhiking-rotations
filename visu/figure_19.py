import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from pathlib import Path
import pandas as pd
from hitchhiking_rotations.utils import RotRep


exps = ["pcd_to_pose", "cube_image_to_pose"]

assert len(sys.argv) == 2, "Please provide the experiment name as an argument."
assert sys.argv[1] in exps, f"Experiment name should be one of {exps}"

exp = sys.argv[1]

plt.figure(figsize=(14, 14))
plt.style.use(os.path.join(HITCHHIKING_ROOT_DIR, "assets", "prettyplots.mplstyle"))
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})

for j, selected_metric in enumerate(["geodesic_distance", "chordal_distance"]):
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
        "r9_svd_geodesic_distance": str(RotRep.SVD) + "-Geo",
        "r9_svd_chordal_distance": str(RotRep.SVD) + "-Chordal",
        "r9_svd_l2": str(RotRep.SVD) + "-MSE",
        "r9_svd_l1": str(RotRep.SVD) + "-MAE",
        "      ": "      ",
        "r9_geodesic_distance": str(RotRep.ROTMAT) + "-Geo",
        "r9_chordal_distance": str(RotRep.ROTMAT) + "-Chordal",
        "r9_l2": str(RotRep.ROTMAT) + "-MSE",
        "r9_l1": str(RotRep.ROTMAT) + "-MAE",
        "       ": "       ",
        "r6_gso_geodesic_distance": str(RotRep.GSO) + "-Geo",
        "r6_gso_chordal_distance": str(RotRep.GSO) + "-Chordal",
        "r6_gso_l2": str(RotRep.GSO) + "-MSE",
        "r6_gso_l1": str(RotRep.GSO) + "-MAE",
        "": "",
        "r6_l2": str(RotRep.RSIX) + "-MSE",
        "r6_l1": str(RotRep.RSIX) + "-MAE",
        " ": " ",
        "quat_c_geodesic_distance": str(RotRep.QUAT_C) + "-Geo",
        "quat_c_chordal_distance": str(RotRep.QUAT_C) + "-Chordal",
        "quat_c_cosine_distance": str(RotRep.QUAT_C) + "-CD",
        "quat_c_l2_dp": str(RotRep.QUAT_C) + "-MSE-DP",
        "quat_c_l2": str(RotRep.QUAT_C) + "-MSE",
        "quat_c_l1": str(RotRep.QUAT_C) + "-MAE",
        "  ": "  ",
        # "quat_rf_cosine_distance": str(RotRep.QUAT_RF) + "-CD",
        "quat_rf_l2_dp": str(RotRep.QUAT_RF) + "-MSE-DP",
        # "quat_rf_l2": str(RotRep.QUAT_RF) + "-MSE",
        # "quat_rf_l1": str(RotRep.QUAT_RF) + "-MAE",
        "   ": "   ",
        "rotvec_geodesic_distance": str(RotRep.EXP) + "-Geo",
        "rotvec_chordal_distance": str(RotRep.EXP) + "-Chordal",
        "rotvec_l2": str(RotRep.EXP) + "-MSE",
        "rotvec_l1": str(RotRep.EXP) + "-MAE",
        "    ": "    ",
        "euler_geodesic_distance": str(RotRep.EULER) + "-Geo",
        "euler_chordal_distance": str(RotRep.EULER) + "-Chordal",
        "euler_l2": str(RotRep.EULER) + "-MSE",
        "euler_l1": str(RotRep.EULER) + "-MAE",
    }

    for k, v in mapping.items():
        df.loc[df["method"] == k, "method"] = v

    df["method"] = pd.Categorical(df["method"], categories=[v for v in mapping.values()], ordered=True)

    plt.subplot(1, 2, j + 1)

    sns.boxplot(
        data=df,
        x="score",
        y="method",
        palette="Blues",
        orient="h",
        width=0.5,
        linewidth=1.5,
        fliersize=2.5,
        showfliers=False,
    )

    # plt.xlabel(f"Error - {selected_metric}")

    if j == 0:
        plt.xlabel(f"Error - Geodesic distance")
        print("Warning: Hardcoded label for the first plot. Please check if it is correct.")
        print(f"Geodesic distance VS {selected_metric}")
    elif j == 1:
        plt.xlabel(f"Error - Chordal distance")
        print("Warning: Hardcoded label for the first plot. Please check if it is correct.")
        print(f"Chordal distance VS {selected_metric}")
    plt.ylabel("")
    # plt.xscale("log")
    plt.tight_layout()

out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", exp, f"figure_19_combined.pdf")

plt.savefig(out_p)

plt.show()
