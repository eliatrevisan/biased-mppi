import torch
import os
from simulator import Simulator
from objective import Objective
from priors import Priors
from dynamics import OmnidirectionalPointRobotDynamics
from mppi_torch.mppi import MPPIPlanner
import yaml
from tqdm import tqdm

abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))


def run_point_robot_example():
    simulator = Simulator(
        cfg=CONFIG["simulator"],
        dt=CONFIG["dt"],
        goal=CONFIG["goal"],
        initial_pose=CONFIG["initial_pose"],
        device=CONFIG["device"],
    )
    dynamics = OmnidirectionalPointRobotDynamics(
        dt=CONFIG["dt"],
        device=CONFIG["device"]
    )
    objective = Objective(
        goal=CONFIG["goal"],
        device=CONFIG["device"]
    )
    priors = Priors(
        cfg=CONFIG
    )
    planner = MPPIPlanner(
        cfg=CONFIG["mppi"],
        nx=6,
        dynamics=dynamics.step,
        running_cost=objective.compute_running_cost,
        prior = priors.compute_priors,
    )

    initial_action = torch.zeros(3, device=CONFIG["device"])
    observation = simulator.step(initial_action)

    for _ in tqdm(range(CONFIG["steps"])):
        action = planner.command(observation)

        observation = simulator.step(action)


if __name__ == "__main__":
    run_point_robot_example()
