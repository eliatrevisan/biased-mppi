from urdfenvs.robots.generic_urdf import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from urdfenvs.sensors.full_sensor import FullSensor
import gymnasium as gym
import torch
import numpy as np
import pybullet as p

class Simulator:
    def __init__(self, cfg, dt, goal, initial_pose, device) -> None:
        self._device = device
        self._goal = goal
        self._initial_pose = initial_pose
        self._dt = dt
        self._environment = self._initalize_environment(cfg)

    def _initalize_environment(self, cfg) -> UrdfEnv:
        """
        Initializes the simulation environment.

        Adds an obstacle and goal visualizaion to the environment and
        steps the simulation once.

        Params
        ----------
        render
            Boolean toggle to set rendering on (True) or off (False).
        """
        robots = [
            GenericDiffDriveRobot(
                urdf=cfg["urdf"],
                mode=cfg["mode"],
                actuated_wheels=[
                    "rear_right_wheel",
                    "rear_left_wheel",
                    "front_right_wheel",
                    "front_left_wheel",
                ],
                castor_wheels=[],
                wheel_radius = 0.098,
                wheel_distance = 2 * 0.187795 + 0.08,
        ),
    ]
        env: UrdfEnv = UrdfEnv(dt=self._dt, robots=robots, render=cfg['render'], observation_checking=False)

        p.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=0,
        cameraPitch=-91,
        cameraTargetPosition=[-1, 0, 0],
        )

        # Set the initial position and velocity of the robot
        env.reset(pos=np.array(self._initial_pose))
        
        # Add the goal
        goal_dict = {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 1,
            "desired_position": self._goal,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
        goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
        env.add_goal(goal)

        # Add the obstacle
        obst1Dict = {
            "type": "sphere",
            "geometry": {
                # "trajectory": ["0.0", "2.0 - sp.Piecewise((5 * (t-3.05) , t > 3.05), (0, True))", "0.0"],
                # "trajectory": ["0.0", "sp.Piecewise((0.0 , t > 3.05), (2.0, True))", "0.0"],
                "trajectory": ["0.0", "sp.Piecewise((2.0 - 2*(t - 2.2), sp.And(t > 2.2, t < 3.2)), (0.0 , t > 3.2), (2.0, True))", "0.0"],
                # "trajectory": ["0.0", "sp.Piecewise((2.0 - 2*(t - 2.2), t > 2.2), (2.0, True))", "0.0"],
                "radius": 0.5,
            },
            # light brown color
            "rgba": [0.7, 0.5, 0.3, 1.0],
        } 
        boxObst = DynamicSphereObstacle(name="simpleBox", content_dict=obst1Dict)
        env.add_obstacle(boxObst)

        # Add the sensor
        sensor = FullSensor(
            goal_mask=["position"],
            obstacle_mask=["position", "velocity", "size"],
            variance=0.0,
        )
        env.add_sensor(sensor, [0])
        env.set_spaces()

        return env

    # def step(self, action: torch.Tensor) -> torch.Tensor:
    #     observation_dict, _, terminated, _, info = self._environment.step(action)
    #     observation_tensor = torch.tensor(
    #         [
    #             [*observation_dict["robot_0"]["joint_state"]["position"],
    #             *observation_dict["robot_0"]["joint_state"]["velocity"]]
    #         ],
    #         device=self._device,
    #     )
    #     return observation_tensor