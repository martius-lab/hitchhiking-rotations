def get_cfg_pcd_to_pose(device):
    shared_trainer_cfg = {
        "_target_": "hitchhiking_rotations.utils.Trainer",
        "lr": 0.001,
        "optimizer": "Adam",
        "logger": "${logger}",
        "verbose": "${verbose}",
        "preprocess_input": "${u:flatten}",
        "device": device,
    }

    return {
        "verbose": False,
        "batch_size": 32,
        "epochs": 100,
        "training_data": {
            "_target_": "hitchhiking_rotations.datasets.PointCloudDataset",
            "mode": "train",
            "device": device,
        },
        "test_data": {
            "_target_": "hitchhiking_rotations.datasets.PointCloudDataset",
            "mode": "test",
            "device": device,
        },
        "validation_data": {
            "_target_": "hitchhiking_rotations.datasets.PointCloudDataset",
            "mode": "val",
            "device": device,
        },
        "model_9": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 9},
        "model_6": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 6},
        "model_4": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 4},
        "model_3": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 3},
        "logger": {
            "_target_": "hitchhiking_rotations.utils.OrientationLogger",
            "metrics": ["l1", "l2", "geodesic_distance", "chordal_distance"],
        },
        "trainers": {
            "r9_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model_9}",
                },
            },
        },
    }
