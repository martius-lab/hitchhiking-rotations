#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
def get_cfg_pose_to_fourier(device, nb, nf):
    cfg = {
        "_target_": "hitchhiking_rotations.utils.Trainer",
        "lr": 0.001,
        "optimizer": "Adam",
        "logger": "${logger}",
        "verbose": "${verbose}",
        "device": device,
        "preprocess_target": "${u:passthrough}",
        "postprocess_pred_loss": "${u:passthrough}",
        "postprocess_pred_logging": "${u:passthrough}",
        "loss": "${u:l2}",
    }

    return {
        "verbose": False,
        "batch_size": 64,
        "epochs": 500,
        "training_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToFourierDataset",
            "mode": "train",
            "dataset_size": 800,
            "device": device,
            "nb": nb,
            "nf": nf,
        },
        "test_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToFourierDataset",
            "mode": "test",
            "dataset_size": 1000,
            "device": device,
            "nb": nb,
            "nf": nf,
        },
        "val_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToFourierDataset",
            "mode": "val",
            "dataset_size": 200,
            "device": device,
            "nb": nb,
            "nf": nf,
        },
        "model9": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 9, "output_dim": 1},
        "model6": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 6, "output_dim": 1},
        "model4": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 4, "output_dim": 1},
        "model3": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 3, "output_dim": 1},
        "logger": {
            "_target_": "hitchhiking_rotations.utils.OrientationLogger",
            "metrics": ["l2"],
        },
        "trainers": {
            "r9_l2": {**cfg, **{"preprocess_input": "${u:flatten}", "model": "${model9}"}},
            "r6_l2": {**cfg, **{"preprocess_input": "${u:rotmat_to_gramschmidt_f}", "model": "${model6}"}},
            "quat_aug_l2": {**cfg, **{"preprocess_input": "${u:rotmat_to_quaternion_aug}", "model": "${model4}"}},
            "quat_c_l2": {**cfg, **{"preprocess_input": "${u:rotmat_to_quaternion_canonical}", "model": "${model4}"}},
            "quat_rf_l2": {**cfg, **{"preprocess_input": "${u:rotmat_to_quaternion_rand_flip}", "model": "${model4}"}},
            "euler_l2": {**cfg, **{"preprocess_input": "${u:rotmat_to_euler}", "model": "${model3}"}},
            "rotvec_l2": {**cfg, **{"preprocess_input": "${u:rotmat_to_rotvec}", "model": "${model3}"}},
        },
    }