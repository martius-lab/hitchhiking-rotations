def get_cfg_pose_to_cube_image(device):
    cfg = {
        "_target_": "hitchhiking_rotations.utils.Trainer",
        "lr": 0.001,
        "optimizer": "SGD",
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
        "batch_size": 32,
        "epochs": 5,
        "training_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToCubeImageDataset",
            "mode": "train",
            "dataset_size": 2048,
            "device": device,
        },
        "test_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToCubeImageDataset",
            "mode": "test",
            "dataset_size": 2048,
            "device": device,
        },
        "val_data": {
            "_target_": "hitchhiking_rotations.datasets.PoseToCubeImageDataset",
            "mode": "val",
            "dataset_size": 2048,
            "device": device,
        },
        "model9": {"_target_": "hitchhiking_rotations.models.CNN", "width": 32, "height": 32, "input_dim": 9},
        "model6": {"_target_": "hitchhiking_rotations.models.CNN", "width": 32, "height": 32, "input_dim": 6},
        "model4": {"_target_": "hitchhiking_rotations.models.CNN", "width": 32, "height": 32, "input_dim": 4},
        "model3": {"_target_": "hitchhiking_rotations.models.CNN", "width": 32, "height": 32, "input_dim": 3},
        "logger": {
            "_target_": "hitchhiking_rotations.utils.OrientationLogger",
            "metrics": ["l2"],
        },
        "trainers": {
            "r9": {**cfg, **{"preprocess_input": "${u:flatten}", "model": "${model9}"}},
            "r6": {**cfg, **{"preprocess_input": "${u:rotmat_to_gramschmidt_f}", "model": "${model6}"}},
            "quat_c": {**cfg, **{"preprocess_input": "${u:rotmat_to_quaternion_canonical}", "model": "${model4}"}},
            "quat_rf": {**cfg, **{"preprocess_input": "${u:rotmat_to_quaternion_rand_flip}", "model": "${model4}"}},
            "euler": {**cfg, **{"preprocess_input": "${u:rotmat_to_euler}", "model": "${model3}"}},
            "rotvec": {**cfg, **{"preprocess_input": "${u:rotmat_to_rotvec}", "model": "${model3}"}},
        },
    }
