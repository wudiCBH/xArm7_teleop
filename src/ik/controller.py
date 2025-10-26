from pathlib import Path
import numpy as np
from pytransform3d.rotations import quaternion_from_matrix, matrix_from_quaternion
from .motion_control import PinocchioMotionControl
from .filters import LPFilter, LPRotationFilter

def wrist_filters(mat, pos_filter, rot_filter):
    filtered = mat.copy()
    filtered[:3, 3] = pos_filter.next(mat[:3, 3])
    filtered[:3, :3] = matrix_from_quaternion(
        rot_filter.next(quaternion_from_matrix(mat[:3, :3]))
    )
    return filtered

class XArm7Controller:
    def __init__(self, cfg: dict, debug: bool = False):
        self.debug = debug
        self.cfg = cfg
        self.dof = cfg["dof_num"]
        self.joint_names = cfg["joint_names"]
        self._qpos = np.zeros(self.dof)
        self.arm_indices = cfg["arm_indices"]

        urdf_path = Path(cfg["urdf_path"])
        self.ee_controller = PinocchioMotionControl(
            urdf_path,
            cfg["ee"],
            np.array(cfg["arm_init"], dtype=float),
            cfg["arm"],
            arm_indices=self.arm_indices,
            debug=debug,
        )

        alpha = cfg["wrist_low_pass_alpha"]
        self.pos_filter = LPFilter(alpha)
        self.rot_filter = LPRotationFilter(alpha)

    def update(self, wrist_T):
        wrist_T = wrist_filters(wrist_T, self.pos_filter, self.rot_filter)
        q = self.ee_controller.control(wrist_T[:3, 3], wrist_T[:3, :3])
        self._qpos[self.arm_indices] = q

    @property
    def qpos(self):
        return self._qpos.astype(np.float32).copy()
