#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
def get_cfg_pcd_to_pose(device):
    shared_trainer_cfg = {
        "_target_": "hitchhiking_rotations.utils.Trainer",
        "lr": 0.001,
        "optimizer": "Adam",
        "logger": "${logger}",
        "verbose": "${verbose}",
        "preprocess_input": "${u:passthrough}",
        "device": device,
    }

    return {
        "verbose": False,
        "batch_size": 32,
        "epochs": 300,
        "training_data": {
            "_target_": "hitchhiking_rotations.datasets.PointCloudDataset",
            "mode": "train",
            "device": device,
        },
        "val_data": {
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
        "model9": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 9},
        "model6": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 6},
        "model4": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 4},
        "model3": {"_target_": "hitchhiking_rotations.models.MLPNetPCD", "in_size": (6, 3000), "out_size": 3},
        "logger": {
            "_target_": "hitchhiking_rotations.utils.OrientationLogger",
            "metrics": ["l1", "l2", "geodesic_distance", "chordal_distance"],
        },
        "trainers": {
            "r9_svd_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model9}",
                },
            },
            "r9_svd_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model9}",
                },
            },
            "r9_svd_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model9}",
                },
            },
            "r9_svd_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:procrustes_to_rotmat}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model9}",
                },
            },
            "r9_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:flatten}",
                    "postprocess_pred_loss": "${u:flatten}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model9}",
                },
            },
            "r9_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:flatten}",
                    "postprocess_pred_loss": "${u:flatten}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model9}",
                },
            },
            "r9_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:n_3x3}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model9}",
                },
            },
            "r9_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:n_3x3}",
                    "postprocess_pred_logging": "${u:procrustes_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model9}",
                },
            },
            "r6_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_gramschmidt_f}",
                    "postprocess_pred_loss": "${u:flatten}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model6}",
                },
            },
            "r6_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_gramschmidt_f}",
                    "postprocess_pred_loss": "${u:flatten}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model6}",
                },
            },
            "r6_gso_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model6}",
                },
            },
            "r6_gso_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model6}",
                },
            },
            "r6_gso_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model6}",
                },
            },
            "r6_gso_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:gramschmidt_to_rotmat}",
                    "postprocess_pred_logging": "${u:gramschmidt_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model6}",
                },
            },
            "quat_c_geodesic_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:quaternion_to_rotmat}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:geodesic_distance}",
                    "model": "${model4}",
                },
            },
            "quat_c_chordal_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:passthrough}",
                    "postprocess_pred_loss": "${u:quaternion_to_rotmat}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:chordal_distance}",
                    "model": "${model4}",
                },
            },
            "quat_c_cosine_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_canonical}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:cosine_distance}",
                    "model": "${model4}",
                },
            },
            "quat_c_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_canonical}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model4}",
                },
            },
            "quat_c_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_canonical}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model4}",
                },
            },
            "quat_c_l2_dp": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_canonical}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l2_dp}",
                    "model": "${model4}",
                },
            },
            "quat_rf_cosine_distance": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_rand_flip}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:cosine_distance}",
                    "model": "${model4}",
                },
            },
            "quat_rf_l2": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_rand_flip}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l2}",
                    "model": "${model4}",
                },
            },
            "quat_rf_l1": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_rand_flip}",
                    "postprocess_pred_loss": "${u:passthrough}",
                    "postprocess_pred_logging": "${u:quaternion_to_rotmat}",
                    "loss": "${u:l1}",
                    "model": "${model4}",
                },
            },
            "quat_rf_l2_dp": {
                **shared_trainer_cfg,
                **{
                    "preprocess_target": "${u:rotmat_to_quaternion_rand_flip}",
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
