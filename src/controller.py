from pathlib import Path
from typing import List, Dict
import numpy as np
from pytransform3d.rotations import quaternion_from_matrix, matrix_from_quaternion

from motion_control import PinocchioMotionControl
from filters import LPFilter, LPRotationFilter

def wrist_filters(
    wrist_mat: np.ndarray, pos_filter: LPFilter, rot_filter: LPRotationFilter
) -> np.ndarray:
    filtered_mat = wrist_mat.copy()
    filtered_mat[:3, 3] = pos_filter.next(wrist_mat[:3, 3])
    filtered_mat[:3, :3] = matrix_from_quaternion(
        rot_filter.next(quaternion_from_matrix(wrist_mat[:3, :3]))
    )
    return filtered_mat

class xArm7Controller:
    def __init__(self, cfg: dict = None, debug: bool = False) -> None:
        if cfg is None:
            raise ValueError("Need to provide a config file.")

        src_dir = Path(__file__).resolve().parent
        project_root = src_dir.parent
        self.default_urdf_dir = project_root / "assets"
        self.debug = debug

        self.dof = cfg["dof_num"]
        self._qpos = np.zeros(self.dof)
        self.cfg = cfg
        
        self._init_indices()
        self._init_controllers()
        self._init_filters()

    def _init_indices(self) -> None:
        self.arm_indices = self.cfg["arm_indices"]

    def _init_controllers(self) -> None:
        urdf_path = self.default_urdf_dir / Path(self.cfg["urdf_path"])
        arm_config = self.cfg["arm"]

        self.ee_controller = PinocchioMotionControl(
            urdf_path,
            self.cfg["ee"],
            np.array(self.cfg['arm_init']),
            arm_config,
            arm_indices=self.arm_indices,
            debug = self.debug
        )

        self.configured = True

    def _init_filters(self) -> None:
        wrist_alpha = self.cfg["wrist_low_pass_alpha"]

        self.wrist_pos_filter = LPFilter(wrist_alpha)
        self.wrist_rot_filter = LPRotationFilter(wrist_alpha)

    def update(
        self,
        wrist: np.ndarray,
    ) -> None:
        if not self.configured:
            raise ValueError("Xarm controller has not been configured.")

        wrist = wrist_filters(
            wrist, self.wrist_pos_filter, self.wrist_rot_filter
        )

        if self.debug:
            print(f"[Debug] Wrist after filter is {wrist}")

        arm_qpos = self.ee_controller.control(
            wrist[:3, 3], wrist[:3, :3]
        )

        self._qpos[self.arm_indices] = arm_qpos

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos.astype(np.float32).copy()
