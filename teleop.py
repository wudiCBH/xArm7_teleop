import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path
import yaml
import pytransform3d.rotations as prt
import pytransform3d.transformations as ptf

from controller import xArm7Controller

def load_config(config_file_name: str) -> Dict[str, Any]:
    robot_config_path = (
        f"." / Path(config_file_name)
    )
    with Path(robot_config_path).open("r") as f:
        cfg = yaml.safe_load(f)["robot_cfg"]

    return cfg

class xArm7Teleop:
    def __init__(self, cfg: Dict[str, Any], debug: bool = False):
        self.controller = xArm7Controller(cfg, debug)
        self.debug = debug

    def step(self,target_ee_pose) -> Tuple[List[float], Any]:

        wrist_Rotation_matrix = prt.matrix_from_euler(target_ee_pose[:3], 0, 1, 2, False)
        wrist_Position = target_ee_pose[-3:]
        wrist_target_ee_matrix = ptf.transform_from(wrist_Rotation_matrix, wrist_Position)
        if self.debug:
            print(f"[Debug] Wrist_target_ee_matrix: {wrist_target_ee_matrix}")

        self.controller.update(wrist_target_ee_matrix)
        cmd = self.controller.qpos

        return cmd


def main():
    config_file_name = "./xarm7.yml"
    cfg = load_config(config_file_name)
    debug = True

    if debug:
        print(f"[INFO] cfg name is {cfg['name']}")
        print("[INFO] Load cfg successfully!!!")

    teleop = xArm7Teleop(cfg, debug)

    if debug:
        print("[INFO] Initialize xArm7Teleop successfully!!!")

    target_ee_pose = np.array([np.deg2rad(180.0), np.deg2rad(0.0), np.deg2rad(0.0), 0.206, 0.0, 0.1205])
    cmd = teleop.step(target_ee_pose)

    if debug:
        print(f"[INFO] target ee pose is {target_ee_pose}")
        print(f"[INFO] final cmd is {cmd}")

if __name__ == "__main__":
    main()