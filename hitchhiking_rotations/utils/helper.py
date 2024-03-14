#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
def passthrough(*x):
    if len(x) == 1:
        return x[0]
    return x


def flatten(x):
    return x.reshape(x.shape[0], -1)


def n_3x3(x):
    return x.reshape(-1, 3, 3)
