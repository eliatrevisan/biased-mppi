import torch


class Objective(object):
    def __init__(self, goal=[1.0, 1.0], device="cuda:0"):
        self.nav_goal = torch.tensor(goal, device=device)

    def compute_running_cost(self, state: torch.Tensor):
        robot_pose = state[:, 0:2]
        goal_dist = torch.linalg.norm(robot_pose - self.nav_goal, axis=1)

        obs_pose = state[:, 6:8]
        # Assume the obstacle is a sphere with radius 0.5
        # Give a cost if collision
        obs_dist = torch.linalg.norm(robot_pose - obs_pose, axis=1)
        obs_cost = torch.where(obs_dist < 1, 0.5, 0.0)


        return goal_dist * 1.0 + obs_cost * 100.0
