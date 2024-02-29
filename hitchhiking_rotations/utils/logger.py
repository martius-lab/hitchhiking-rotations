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

    def log(self, mode, epoch, pred, target):
        for metric, res in self.modes[mode].items():
            res["sum"] += available_metrics[metric](pred, target).item()
            res["count"] += 1

    def reset(self):
        for mo in self.modes.keys():
            for res in self.modes[mo].values():
                res["sum"] = 0
                res["count"] = 0
