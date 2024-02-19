from enum import Enum
import numpy as np
import argparse
import torch

def default(*x):
    if len(x) == 1:
        return x[0]
    return x

def svd_orthonormalization(x):
    pass

def gram_schmidt_orthonormalization(x):
    pass

def quat_to_rotmat(x):
    # normalize first
    pass

def axis_angle_to_rotmat(x):
    pass
def euler_to_rotmat(x):
    pass


def quat_aug_dataset(quats: np.ndarray, ixs):
    # quats: (N, M, .., 4)
    # return augmented inputs and quats
    return (np.concatenate((quats, -quats), axis=0), *np.concatenate((ixs, ixs), axis=0))

def quat_hm_dataset(quats: np.ndarray):
    # quats: (N, M, .., 4)
    return np.where(quats[..., 3] < 0, -quats, quats)


# losses

def l1_loss(vec_gt, vec_pred):
    pass

def l2_loss(vec_gt, vec_pred):
    pass

def distance_picking_loss(vec_gt, vec_pred):
    pass

def cosine_distance_loss(vec_gt, vec_pred):
    pass

def geodesic_loss(Rot_gt, Rot_pred):
    pass

def chordial_loss(Rot_gt, Rot_pred):
    pass



class Representation(Enum):
    R9 = "r9"
    R6 = "r6"
    QUAT = "quat"
    QUAT_AUG = "quat_aug"
    QUAT_HM = "quat_halfmap"
    AXIS_ANGLE = "axis_angle"
    EULER = "euler"

    def __str__(self) -> str:
        return self.value

    def get_f_out(self):
        fout_dict = {
            Representation.R9: svd_orthonormalization,
            Representation.R6: gram_schmidt_orthonormalization,
            Representation.QUAT: default,
            Representation.QUAT_AUG: default,
            Representation.QUAT_HM: default,
            Representation.AXIS_ANGLE: default,
            Representation.EULER: default,
        }
        return fout_dict[self]

    @property
    def preprocess(self):
        pre_process_dict = {
            Representation.R9: default,
            Representation.R6: default,
            Representation.QUAT:default,
            Representation.QUAT_AUG: quat_aug_dataset,
            Representation.QUAT_HM: quat_hm_dataset,
            Representation.AXIS_ANGLE: default,
            Representation.EULER: default,
        }
        return pre_process_dict[self]


class TrainingLoss(Enum):
    L1 = "l1"  # mean absolute error
    L2 = "l2"  # mean squared error
    QUAT_DP = "l_quat_dp"
    QUAT_CP = "l_quat_cp"
    CHORDIAL = "chordial_loss"
    GEODESIC = "geodesic_loss"

    def __str__(self) -> str:
        return self.value

def get_loss_and_fout(loss: TrainingLoss, representation: Representation):
    loss_dict = {
        TrainingLoss.L1: l1_loss,
        TrainingLoss.L2: l2_loss,
        TrainingLoss.QUAT_DP: distance_picking_loss,
        TrainingLoss.QUAT_CP: cosine_distance_loss,
        TrainingLoss.CHORDIAL: chordial_loss,
        TrainingLoss.GEODESIC: geodesic_loss,
    }

    f_out = representation.get_f_out()

    if representation in [Representation.R9, Representation.R6]:
        assert loss not in [TrainingLoss.QUAT_DP, TrainingLoss.QUAT_CP], f"Loss {loss} not supported for representation {representation}"
        if loss == TrainingLoss.CHORDIAL:
            print(f"Cordial loss for rotations is the same as L2 loss. Using default L2 loss")
            loss = TrainingLoss.L2

    if representation in [Representation.QUAT, Representation.QUAT_AUG, Representation.QUAT_HM]:
        if loss == TrainingLoss.CHORDIAL or loss == TrainingLoss.GEODESIC:
            f_out = quat_to_rotmat

    if representation == Representation.AXIS_ANGLE:
        if loss == TrainingLoss.CHORDIAL or loss == TrainingLoss.GEODESIC:
            f_out = axis_angle_to_rotmat

    if representation == Representation.EULER:
        if loss == TrainingLoss.CHORDIAL or loss == TrainingLoss.GEODESIC:
            f_out = euler_to_rotmat

    return  loss_dict[loss], f_out





# TEST
print(get_loss_and_fout(TrainingLoss.L1, Representation.R9) == (l1_loss, svd_orthonormalization))
print(get_loss_and_fout(TrainingLoss.L2, Representation.R9) == (l2_loss, svd_orthonormalization))
print(get_loss_and_fout(TrainingLoss.L1, Representation.R6) == (l1_loss, gram_schmidt_orthonormalization))
print(get_loss_and_fout(TrainingLoss.L2, Representation.R6) == (l2_loss, gram_schmidt_orthonormalization))
print(get_loss_and_fout(TrainingLoss.L2, Representation.QUAT) == (l2_loss, default))
print(get_loss_and_fout(TrainingLoss.QUAT_DP, Representation.QUAT) == (distance_picking_loss, default))
print(get_loss_and_fout(TrainingLoss.QUAT_CP, Representation.QUAT_AUG) == (cosine_distance_loss, default))
print(get_loss_and_fout(TrainingLoss.CHORDIAL, Representation.QUAT) == (chordial_loss, quat_to_rotmat))
print(get_loss_and_fout(TrainingLoss.CHORDIAL, Representation.AXIS_ANGLE) == (chordial_loss, axis_angle_to_rotmat))
print(get_loss_and_fout(TrainingLoss.GEODESIC, Representation.EULER) == (geodesic_loss, euler_to_rotmat))

