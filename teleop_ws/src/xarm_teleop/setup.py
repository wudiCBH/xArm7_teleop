from setuptools import setup, find_packages
from pathlib import Path

package_name = "xarm_teleop"
here = Path(__file__).parent
install_requires = [
    "numpy",
    "pytransform3d>=3.4",
    "pin"
]

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["xarm_teleop/config/xarm7.yaml"]),
        ("share/" + package_name + "/launch", ["launch/teleop_cartesian.launch.py"]),
    ],
    install_requires=["setuptools"] + install_requires,
    zip_safe=True,
    maintainer="Revanth Senthilkumaran",
    maintainer_email="revanths@andrew.cmu.edu",
    description="Keyboard EE teleop for IK - JointState for XArm7 in Isaac Sim",
    license="MIT",
    entry_points={
        "console_scripts": [
            "keyboard_ee_teleop = xarm_teleop.nodes.keyboard_ee_teleop:main",
            "ik_to_jointstate = xarm_teleop.nodes.ik_to_jointstate:main",
        ],
    },
)
