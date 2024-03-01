#                                                                               
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
from .metrics import *

available_metrics = {
    "chordal_distance": chordal_distance,
    "cosine_distance": cosine_distance,
    "cosine_similarity": cosine_similarity,
    "geodesic_distance": geodesic_distance,
    "l1": l1,
    "l2": l2,
}


class OrientationLogger:
    def __init__(self, metrics, modes=["train", "val", "test"]):
        self.modes = {}
        for mo in modes:
            self.modes[mo] = {}
            for me in metrics:
                self.modes[mo][me] = {"count": 0, "sum": 0}
            self.modes[mo]["loss"] = {"count": 0, "sum": 0}

        self.metrics = metrics

    def log(self, mode, epoch, pred, target, loss):
        # Always log loss
        self.modes[mode]["loss"]["sum"] += loss
        self.modes[mode]["loss"]["count"] += 1

        for metric in self.metrics:
            res = self.modes[mode][metric]
            res["sum"] += available_metrics[metric](pred, target).item()
            res["count"] += 1

    def reset(self):
        for mo in self.modes.keys():
            for res in self.modes[mo].values():
                res["sum"] = 0
                res["count"] = 0

    def get_score(self, mode, metric):
        m = self.modes[mode][metric]
        return m["sum"] / m["count"]
