import torch

class Priors(object):
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.nav_goal = torch.tensor(cfg["goal"], device=self.device)
        self.horizon = cfg["mppi"]["horizon"]

    def compute_priors(self, state: torch.Tensor, t: int):
        p_ctrl = self.proportional_controller(state)
        diag = torch.tensor([0.7, -0.5, 0.0], device=self.device)
        return torch.stack([p_ctrl, diag])
    
    def proportional_controller(self, state: torch.Tensor):
        # state 0 because the first sample is the p controller!
        positions = state[0, 0:2]
        goal = self.nav_goal
        error = goal - positions

        input = torch.clamp(2.0*error, torch.tensor([-1.0, -1.0], device=self.device), torch.tensor([1.0, 1.0], device=self.device))
        input_with_zero_rot = torch.cat((input, torch.tensor([0.], device=self.device)))
        
        return input_with_zero_rot