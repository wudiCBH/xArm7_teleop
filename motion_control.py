import numpy as np
import pinocchio as pin
from numpy.linalg import norm, solve
from typing import List, Optional, Dict

class PinocchioMotionControl:
    """
    x = T (q_pos)
    v = x_dot = J(q) * q_dot

    error = x_desired * x_inverse
    v_desired = K * error  # we could prove its stability through e_dot = - v = - K * e , lie group & lie algebra
    q_dot_desried = J(q)_inverse * v

    for i in range(iter):
        q = q + q_dot_desired * dt
        x = T (q)
        error = x_desired * x_inverse

        if e = < eps:
            break
    """

    def __init__(
        self,
        urdf_path: str,
        wrist_name: str,
        arm_init_qpos: np.ndarray,
        arm_config: Dict[str, float],
        arm_indices: Optional[List[int]] = [],
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.arm_indices = arm_indices
        self.alpha = float(arm_config["out_lp_alpha"])

        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.dof = self.model.nq

        if arm_indices:
            locked_joint_ids = list(set(range(self.dof)) - set(self.arm_indices))
            locked_joint_ids = [
                id + 1 for id in locked_joint_ids
            ]  # account for universe joint
            self.model = pin.buildReducedModel(
                self.model, locked_joint_ids, np.zeros(self.dof)
            )
        self.arm_dof = self.model.nq

        self.lower_limit = self.model.lowerPositionLimit
        self.upper_limit = self.model.upperPositionLimit
        self.data: pin.Data = self.model.createData()

        self.wrist_id = self.model.getFrameId(wrist_name)

        # print(arm_init_qpos)
        self.qpos = arm_init_qpos
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.wrist_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.wrist_id
        )

        if self.debug:
            print(f"[Debug] Initial Wrist Pose Matrix is {self.wrist_pose}")

        self.damp = float(arm_config["damp"])
        self.ik_eps = float(arm_config["eps"])
        self.dt = float(arm_config["dt"])

    def control(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        oMdes = pin.SE3(target_rot, target_pos)
        qpos = self.qpos.copy()

        ik_qpos = qpos.copy()
        ik_qpos = self.ik_clik(ik_qpos, oMdes, self.wrist_id)
        qpos = ik_qpos.copy()

        self.qpos = pin.interpolate(self.model, self.qpos, qpos, self.alpha)
        self.qpos = qpos.copy()

        return self.qpos.copy()

    def ik_clik(
        self, qpos: np.ndarray, oMdes: pin.SE3, wrist_id: int, iter: int = 1000
    ) -> np.ndarray:
        for _ in range(iter):
            pin.forwardKinematics(self.model, self.data, qpos)

            wrist_pose = pin.updateFramePlacement(self.model, self.data, wrist_id)
            iMd = wrist_pose.actInv(oMdes)

            err = pin.log(iMd).vector
            if norm(err) < self.ik_eps:
                break

            J = pin.computeFrameJacobian(self.model, self.data, qpos, wrist_id)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            qpos = pin.integrate(self.model, qpos, v * self.dt)

        qpos = np.clip(qpos, self.lower_limit, self.upper_limit)
        return qpos
