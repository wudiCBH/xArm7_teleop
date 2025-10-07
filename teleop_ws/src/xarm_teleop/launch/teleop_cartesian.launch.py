from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    domain = DeclareLaunchArgument("ros_domain_id", default_value="0")
    cfg = DeclareLaunchArgument("config_path", default_value="")
    urdf = DeclareLaunchArgument("urdf_path", default_value="")
    frame = DeclareLaunchArgument("frame_id", default_value="world")
    rate = DeclareLaunchArgument("rate_hz", default_value="50.0")
    debug = DeclareLaunchArgument("debug", default_value="false")

    return LaunchDescription([
        domain, cfg, urdf, frame, rate, debug,

        Node(
            package="xarm_teleop",
            executable="keyboard_ee_teleop",
            name="keyboard_ee_teleop",
            parameters=[{
                "frame_id": LaunchConfiguration("frame_id"),
                "rate_hz": LaunchConfiguration("rate_hz"),
            }],
            output="screen",
        ),
        Node(
            package="xarm_teleop",
            executable="ik_to_jointstate",
            name="ik_to_jointstate",
            parameters=[{
                "config_path": LaunchConfiguration("config_path"),
                "urdf_path": LaunchConfiguration("urdf_path"),
                "debug": LaunchConfiguration("debug"),
            }],
            output="screen",
        ),
    ])
