import numpy as np


class TowerState:
    def __init__(self, model, data, layers=9, blocks_per_layer=3):
        self.model = model
        self.data = data
        self.block_names = [
            f"b{layer}_{pos}"
            for layer in range(1, layers + 1)
            for pos in range(1, blocks_per_layer + 1)
        ]
        # Reference positions captured once the tower has settled.
        self.initial_pos = {}

    def record_initial(self):
        for name in self.block_names:
            self.initial_pos[name] = self.data.body(name).xpos.copy()

    # --- per-block measurements (all relative to the settled snapshot) --------

    def slide(self, name):
        dx, dy, _ = self.data.body(name).xpos - self.initial_pos[name]
        return float(np.hypot(dx, dy))

    def drop(self, name):
        return float(self.initial_pos[name][2] - self.data.body(name).xpos[2])

    def tilt(self, name):
        # 3rd column of the body rotation matrix = block's local z in world frame.
        up_z = self.data.body(name).xmat.reshape(3, 3)[2, 2]
        return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))

    # --- whole-tower summaries ------------------------------------------------

    def _others(self, exclude):
        if exclude is None:
            return self.block_names
        exclude = {exclude} if isinstance(exclude, str) else set(exclude)
        return [n for n in self.block_names if n not in exclude]

    def max_drop(self, exclude=None):
        return max(self.drop(n) for n in self._others(exclude))

    def max_tilt(self, exclude=None):
        return max(self.tilt(n) for n in self._others(exclude))

    def is_settled(self, vel_thresh=0.02):
        return float(np.abs(self.data.qvel).max()) < vel_thresh

    def is_collapsed(self, drop_thresh=0.015, tilt_thresh=20.0, exclude=None):
        return (
            self.max_drop(exclude) > drop_thresh
            or self.max_tilt(exclude) > tilt_thresh
        )

    def summary(self):
        worst_slide = max(self.block_names, key=self.slide)
        return {
            "max_slide": (worst_slide, round(self.slide(worst_slide), 4)),
            "max_drop": round(self.max_drop(), 4),
            "max_tilt": round(self.max_tilt(), 2),
            "collapsed": self.is_collapsed(),
        }
