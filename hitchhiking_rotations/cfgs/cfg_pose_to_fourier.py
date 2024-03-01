#                                                                               
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
def get_cfg_pose_to_fourier(device, nb, nf):
    shared_trainer_cfg = {
        "_target_": "hitchhiking_rotations.utils.Trainer",
        "lr": 0.001,
        "optimizer": "SGD",
        "logger": "${logger}",
        "verbose": "${verbose}",
        "device": device,
    }

    return {
        "verbose": False,
        "batch_size": 32,
        "epochs": 5,
        "training_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToFourierDataset",
            "mode": "train",
            "nb": nb,
            "nf": nf,
            "device": device,
        },
        "test_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToFourierDataset",
            "mode": "test",
            "nb": nb,
            "nf": nf,
            "device": device,
        },
        "val_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToFourierDataset",
            "mode": "val",
            "nb": nb,
            "nf": nf,
            "device": device,
        },
        "model9": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 9},
        # Maybe here we have to also change the logger - but the l2 metric may do it
        "logger": {
            "_target_": "hitchhiking_rotations.utils.OrientationLogger",
            "metrics": ["l2"],
        },
        "trainers": {
            "r9_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_input": "${u:flatten}",
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model9}",
                },
            }
        },
    }
