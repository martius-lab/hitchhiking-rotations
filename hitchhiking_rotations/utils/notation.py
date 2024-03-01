#                                                                               
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
from enum import Enum


class RotRep(Enum):
    GSO = "$\mathbb{R}^6$+GSO"
    SVD = "$\mathbb{R}^9$+SVD"
    QUAT_C = "Quat$^+$"
    QUAT = "Quat"
    EULER = "Euler"
    EXP = "Exp"

    def __str__(self):
        return "%s" % self.value
