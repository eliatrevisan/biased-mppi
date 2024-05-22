import torch
import os
from simulator import Simulator
from objective import Objective
from dynamics import JackalDynamics
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
        initial_pose=CONFIG["initial_actor_positions"],
        device=CONFIG["device"],
    )
    dynamics = JackalDynamics(
        dt=CONFIG["dt"], device=CONFIG["device"]
    )
    objective = Objective(goal=CONFIG["goal"], device=CONFIG["device"])
    planner = MPPIPlanner(
        cfg=CONFIG["mppi"],
        nx=12,
        dynamics=dynamics.step,
        running_cost=objective.compute_running_cost,
    )

    initial_action = torch.zeros(2, device=CONFIG["device"])
    ob, *_ =  simulator._environment.step(initial_action.cpu().numpy())

    # simulator._environment.start_video_recording("jackal_nobias_sim.mp4")

    for _ in tqdm(range(CONFIG["steps"])):
        ob_robot = ob["robot_0"]
        obst = ob["robot_0"]["FullSensor"]["obstacles"]
        observation_tensor = torch.tensor(
            [
                [*ob_robot["joint_state"]["position"],
                *ob_robot["joint_state"]["velocity"],
                *obst[3]['position'],
                *obst[3]['velocity']]
            ],
            device=CONFIG["device"],
        )

        action = planner.command(observation_tensor)
        (
            ob,
            *_,
        ) = simulator._environment.step(action.cpu().numpy())

    


if __name__ == "__main__":
    run_point_robot_example()
