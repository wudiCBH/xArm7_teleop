#xRAM7_Sim.py

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a xArm7 to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--debuginfo", type=bool, default=False, help="Whether to print important debug information")
parser.add_argument("--myinfo", type=bool, default=True, help="Whether to print important information")
parser.add_argument("--configfile", type=str, default="xarm7.yml", help="Yaml file for configuration")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
debug = args_cli.debuginfo
info = args_cli.myinfo

# We have to import it at the very beginning because of some deeper bug of IsaacLab and pinocchio
# Initialize the xArm7Teleop
from teleop import xArm7Teleop, load_config
config_file_name = args_cli.configfile
cfg = load_config(config_file_name)
Teleop = xArm7Teleop(cfg, debug)

if info:
    print("[INFO] Initialize Teleop successfully!")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import math
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

ISAAC_NUCLEUS_DIR = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac'

#Add Robot and Scene to IsaacSim
robot_asset_CONFIG = AssetBaseCfg(
    prim_path = "{ENV_REGEX_NS}/xArm7",
    spawn = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Ufactory/xarm7/xarm7.usd"),
)

xARM_7_CONFIG = ArticulationCfg(
    prim_path = "{ENV_REGEX_NS}/xArm7/root_joint",
    spawn=None,
    actuators={"arm_actuators": ImplicitActuatorCfg(joint_names_expr=["joint[1-7].*", "drive_joint"], damping=None, stiffness=None)},
)  

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot = robot_asset_CONFIG
    xArm7 = xARM_7_CONFIG

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, teleop: xArm7Teleop, debug: bool, info: bool):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    account = 0
    circle_angular_velocity = 0.3
    ee_cur_pose_list = []
    circle_pos_list = []
    time_list = []

    while simulation_app.is_running():
        
        #Define a circle target
        radius = 0.1
        center = [0.306, 0.0, 0.1205]
        circle_pos = [center[0] + radius * np.cos(circle_angular_velocity * sim_time + np.pi), \
                    center[1] + radius * np.sin(circle_angular_velocity * sim_time + np.pi), center[2]]

        target_ee_pose = np.array([np.deg2rad(180.0), np.deg2rad(0.0), np.deg2rad(0.0)] + circle_pos)

        ##TODO: Add a keyboard controller send target information





        # target_ee_pose = np.array([np.deg2rad(180.0), np.deg2rad(0.0), np.deg2rad(0.0), 0.206, 0.0, 0.1205])
        cmd = teleop.step(target_ee_pose)
        joint_pos = torch.tensor(cmd)

        if debug:
            print(f"[Debug] Command is {joint_pos}")

        scene["xArm7"].write_joint_position_to_sim(joint_pos)
        joint_cur_pos = scene["xArm7"].data.joint_pos.clone()
        ee_cur_pose = scene["xArm7"].data.body_link_state_w.clone()[:,7,:] #(1,15,13),(env_num,link_num,state_dim) the ee link is in the 8th, here state_dim contain[pos,quat,vel,ang_vel]

        if account % 10 == 0:

            ee_cur_pose_list.append(ee_cur_pose.to("cpu"))
            circle_pos_list.append(circle_pos)
            time_list.append(sim_time)
        
        if debug:
            print(f"[Debug] Joint Current Pos is {joint_cur_pos}")
            print(f"[Debug] EE Current Pose is {ee_cur_pose}")

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        account += 1
        scene.update(sim_dt)

        if sim_time >= 30:
            ee_cur_pose_list = np.array(ee_cur_pose_list)
            ee_cur_pose_list = np.squeeze(ee_cur_pose_list, axis=1)
            ee_cur_pose_list = np.transpose(ee_cur_pose_list, (1,0))
            circle_pos_list = np.array(circle_pos_list)
            circle_pos_list = np.transpose(circle_pos_list, (1,0))
            time_list = np.array(time_list)
            if info:
                print(f"[INFO] ee_cur_pose_list shape is {ee_cur_pose_list.shape}, circle_pos_list shape is {circle_pos_list.shape}, time_list shape is {time_list.shape}")

            plt.figure(figsize=(12,4))

            plt.subplot(1,3,1)
            plt.plot(time_list, ee_cur_pose_list[0], color='red', label='ee_pose')
            plt.plot(time_list, circle_pos_list[0], color='blue', label='circle_pos')
            plt.title("X_tracking")
            plt.legend()
            plt.grid(True)

            plt.subplot(1,3,2)
            plt.plot(time_list, ee_cur_pose_list[1], color='red', label='ee_pose')
            plt.plot(time_list, circle_pos_list[1], color='blue', label='circle_pos')
            plt.title("Y_tracking")
            plt.legend()
            plt.grid(True)

            plt.subplot(1,3,3)
            plt.plot(time_list, ee_cur_pose_list[2], color='red', label='ee_pose')
            plt.plot(time_list, circle_pos_list[2], color='blue', label='circle_pos')
            plt.title("Z_tracking")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            break

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    if info:
        print(f"[INFO] ISAAC_NUCLEUS_DIR is {ISAAC_NUCLEUS_DIR}")
        print("[INFO]: Isaac Sim Setup complete...")

    run_simulator(sim, scene, Teleop, debug, info)

if __name__ == "__main__":
    main()
    simulation_app.close()
