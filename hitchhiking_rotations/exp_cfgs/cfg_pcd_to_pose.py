def get_cfg_pcd_to_pose(device):
    shared_trainer_cfg = {
        "_target_": "hitchhiking_rotations.utils.Trainer",
        "lr": 0.001,
        "optimizer": "SGD",
        "logger": "${logger}",
        "verbose": "${verbose}",
        "device": device,
        "preprocess_input": "${u:flatten}",
    }

    return {
        "verbose": True,
        "batch_size": 32,
        "epochs": 5,
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
        "model_pcd_9": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 9},
        "model_pcd_6": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 6},
        "model_pcd_4": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 4},
        "model_pcd_3": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 3},
        "logger": {
            "_target_": "hitchhiking_rotations.utils.OrientationLogger",
            "metrics": ["l1", "l2", "geodesic_distance", "chordal_distance"],
        },
        "trainers": {
            "r9_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_input": "${get_method:hitchhiking_rotations.utils.flatten}",
                    "preprocess_target": "${get_method:hitchhiking_rotations.utils.passthrough}",
                    "postprocess_pred": "${get_method:hitchhiking_rotations.utils.procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${get_method:hitchhiking_rotations.utils.passthrough}",
                    "loss": "${get_method:hitchhiking_rotations.utils.l2}",
                    "model": "${model9}",
                },
            },
        },
    }
