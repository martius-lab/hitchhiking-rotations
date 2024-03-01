def get_cfg_cube_image_to_pose(device):
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
        "verbose": False,
        "batch_size": 32,
        "epochs": 5,
        "training_data": {
            "_target_": "hitchhiking_rotations.datasets.CubeImageToPoseDataset",
            "mode": "train",
            "dataset_size": 2048,
            "device": device,
        },
        "test_data": {
            "_target_": "hitchhiking_rotations.datasets.CubeImageToPoseDataset",
            "mode": "test",
            "dataset_size": 2048,
            "device": device,
        },
        "val_data": {
            "_target_": "hitchhiking_rotations.datasets.CubeImageToPoseDataset",
            "mode": "val",
            "dataset_size": 2048,
            "device": device,
        },
        "model9": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 9},
        "model6": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 6},
        "model4": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 4},
        "model3": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 3},
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
                    "model": "${model9}",
                },
            },
            "r9_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model9}",
                },
            },
            "r9_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model9}",
                },
            },
            "r9_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model9}",
                },
            },
            "r6_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model6}",
                },
            },
            "r6_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model6}",
                },
            },
            "r6_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model6}",
                },
            },
            "r6_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model6}",
                },
            },
            "quat_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:quaternion_to_rotmat}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model4}",
                },
            },
            "quat_cosine_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:cosine_distance}",
                    "model": "${model4}",
                },
            },
            "quat_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model4}",
                },
            },
            "quat_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model4}",
                },
            },
            "quat_l2_dp": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l2_dp}",
                    "model": "${model4}",
                },
            },
            "rotvec_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:rotvec_to_rotmat}",
                    "postprocess_pred_logging": "${u:rotvec_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model3}",
                },
            },
            "rotvec_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:rotvec_to_rotmat}",
                    "postprocess_pred_logging": "${u:rotvec_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model3}",
                },
            },
            "rotvec_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:rotvec_to_rotmat}",
                    "postprocess_pred_logging": "${u:rotvec_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model3}",
                },
            },
            "rotvec_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:rotvec_to_rotmat}",
                    "postprocess_pred_logging": "${u:rotvec_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model3}",
                },
            },
            "euler_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:euler_to_rotmat}",
                    "postprocess_pred_logging": "${u:euler_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model3}",
                },
            },
            "euler_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:euler_to_rotmat}",
                    "postprocess_pred_logging": "${u:euler_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model3}",
                },
            },
            "euler_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:euler_to_rotmat}",
                    "postprocess_pred_logging": "${u:euler_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model3}",
                },
            },
            "euler_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:euler_to_rotmat}",
                    "postprocess_pred_logging": "${u:euler_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model3}",
                },
            },
        },
    }
