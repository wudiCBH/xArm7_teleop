import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import sys, termios, tty, time
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

HELP = """
W/S: ±X,  A/D: ±Y,  R/F: ±Z
I/K: ±Roll,  J/L: ±Pitch,  U/O: ±Yaw
+ / - : increase/decrease step
SPACE: reset pose
Q: quit
"""

def getch(blocking=False):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if not blocking:
            import select
            r, _, _ = select.select([fd], [], [], 0)
            if not r:
                return None
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

class KeyboardEETeleop(Node):
    def __init__(self):
        super().__init__("keyboard_ee_teleop")
        self.pub = self.create_publisher(PoseStamped, "/ee_target", 10)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("start_xyz", [0.306, 0.0, 0.1205])
        self.declare_parameter("start_rpy_deg", [180.0, 0.0, 0.0])
        self.declare_parameter("pos_step", 0.01)
        self.declare_parameter("rot_step_deg", 2.0)

        self.rate = self.get_parameter("rate_hz").value
        self.frame_id = self.get_parameter("frame_id").value
        self.xyz = np.array(self.get_parameter("start_xyz").value, dtype=float)
        self.rpy = np.deg2rad(self.get_parameter("start_rpy_deg").value).astype(float)
        self.pos_step = float(self.get_parameter("pos_step").value)
        self.rot_step = np.deg2rad(float(self.get_parameter("rot_step_deg").value))

        self.get_logger().info(HELP)
        self.timer = self.create_timer(1.0/self.rate, self._tick)

    def _tick(self):
        ch = getch(blocking=False)
        if ch:
            if ch.lower() == 'q':
                rclpy.shutdown()
                return
            elif ch == 'w': self.xyz[0] += self.pos_step
            elif ch == 's': self.xyz[0] -= self.pos_step
            elif ch == 'a': self.xyz[1] += self.pos_step
            elif ch == 'd': self.xyz[1] -= self.pos_step
            elif ch == 'r': self.xyz[2] += self.pos_step
            elif ch == 'f': self.xyz[2] -= self.pos_step
            elif ch == 'i': self.rpy[0] += self.rot_step
            elif ch == 'k': self.rpy[0] -= self.rot_step
            elif ch == 'j': self.rpy[1] += self.rot_step
            elif ch == 'l': self.rpy[1] -= self.rot_step
            elif ch == 'u': self.rpy[2] += self.rot_step
            elif ch == 'o': self.rpy[2] -= self.rot_step
            elif ch == '+': self.pos_step *= 1.2; self.rot_step *= 1.2
            elif ch == '-': self.pos_step /= 1.2; self.rot_step /= 1.2
            elif ch == ' ':
                self.xyz[:] = self.get_parameter("start_xyz").value
                self.rpy[:] = np.deg2rad(self.get_parameter("start_rpy_deg").value)

        R = pr.matrix_from_euler(self.rpy, 0, 1, 2, extrinsic=False)
        T = pt.transform_from(R, self.xyz)

        # publish
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        q = pr.quaternion_from_matrix(R)  # (w,x,y,z)
        # ROS wants (x,y,z,w)
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.xyz.tolist()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q[1], q[2], q[3], q[0]
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = KeyboardEETeleop()
    rclpy.spin(node)
