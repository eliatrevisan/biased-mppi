from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from urdfenvs.sensors.full_sensor import FullSensor
import torch
import numpy as np


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
            GenericUrdfReacher(urdf=cfg["urdf"], mode=cfg["mode"]),
        ]
        env: UrdfEnv = UrdfEnv(dt=self._dt, robots=robots, render=cfg['render'])
        # Set the initial position and velocity of the point mass.
        env.reset(pos=np.array(self._initial_pose))
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
                "position": [0.0, 0.0, 0.0],
                "radius": 0.5,
            },
            # light brown color
            "rgba": [0.7, 0.5, 0.3, 1.0],
        } 
        Obst = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
        env.add_obstacle(Obst)

        # Add the sensor
        sensor = FullSensor(
            goal_mask=["position"],
            obstacle_mask=["position", "velocity", "size"],
            variance=0.0,
        )
        env.add_sensor(sensor, [0])
        env.set_spaces()

        return env

    def step(self, action: torch.Tensor) -> torch.Tensor:
        ob, _, terminated, _, info = self._environment.step(action)
        ob_robot = ob["robot_0"]
        obst = ob["robot_0"]["FullSensor"]["obstacles"]
        observation_tensor = torch.tensor(
            [
                *ob_robot["joint_state"]["position"],
                *ob_robot["joint_state"]["velocity"],
                *obst[3]['position'],
                *obst[3]['velocity']
            ],
            device=self._device,
        )
        return observation_tensor