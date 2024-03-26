#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from enum import Enum


class RotRep(Enum):
    GSO = r"$\mathbb{R}^6$+GSO"
    SVD = r"$\mathbb{R}^9$+SVD"
    QUAT_C = r"Quat$^+$"
    QUAT = r"Quat"
    QUAT_RF = r"Quat$^{\mathrm{RF}}$"
    QUAT_AUG = r"Quat$^{\mathrm{a}+}$"
    EULER = r"Euler"
    EXP = r"Exp"
    ROTMAT = r"$\mathbb{R}^9$"
    RSIX = r"$\mathbb{R}^6$"

    def __str__(self):
        return "%s" % self.value
