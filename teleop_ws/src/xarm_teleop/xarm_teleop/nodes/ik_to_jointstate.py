import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import numpy as np
import yaml
from pathlib import Path
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from xarm_teleop.ik.controller import XArm7Controller

class IKToJointState(Node):
    def __init__(self):
        super().__init__("ik_to_jointstate")

        # params
        self.declare_parameter("config_path", "")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("topic_in", "/ee_target")
        self.declare_parameter("topic_out", "/joint_command")
        self.declare_parameter("debug", False)

        cfg_path = self.get_parameter("config_path").value
        urdf_override = self.get_parameter("urdf_path").value
        self.topic_in = self.get_parameter("topic_in").value
        self.topic_out = self.get_parameter("topic_out").value
        self.debug = bool(self.get_parameter("debug").value)

        if not cfg_path:
            # default to installed config
            share = Path(__file__).resolve().parents[1] / "config" / "xarm7.yaml"
            cfg_path = str(share)

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)["robot_cfg"]

        if urdf_override:
            cfg["urdf_path"] = urdf_override

        self.ctrl = XArm7Controller(cfg, debug=self.debug)
        self.joint_names = cfg["joint_names"]

        self.sub = self.create_subscription(PoseStamped, self.topic_in, self._target_cb, 10)
        self.pub = self.create_publisher(JointState, self.topic_out, 10)

        self.get_logger().info(f"IK ready. Listening on {self.topic_in}, publishing joint commands to {self.topic_out}")

    def _target_cb(self, msg: PoseStamped):
        # convert pose -> SE3
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        # pytransform3d uses (w,x,y,z)
        R = pr.matrix_from_quaternion([qw, qx, qy, qz])
        T = pt.transform_from(R, np.array([x, y, z], dtype=float))

        self.ctrl.update(T)
        q = self.ctrl.qpos  # length 7

        # publish JointState for Isaac ArticulationController via ROS bridge
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = q.tolist()
        self.pub.publish(js)

def main():
    rclpy.init()
    node = IKToJointState()
    rclpy.spin(node)
