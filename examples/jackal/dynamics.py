import torch


class JackalDynamics:
    def __init__(self, dt=0.05, device="cuda:0") -> None:
        self._dt = dt
        self._device = device

    def clip_actions(self, forward_velocity: torch.Tensor, rotational_velocity: torch.Tensor):
        max_linear_vel = 2.0
        max_rot_vel = 1.0
        
        # Clip forward and rotational velocities based on robot constraints
        forward_velocity = torch.clamp(forward_velocity, -max_linear_vel, max_linear_vel)
        rotational_velocity = torch.clamp(rotational_velocity, -max_rot_vel, max_rot_vel)
        return forward_velocity, rotational_velocity
    
    def bycicle_model(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x, y, theta, vx, vy, omega = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]

        actions[:,0], actions[:,1] = self.clip_actions(actions[:,0], actions[:,1])

        # Update velocity and position using bicycle model
        new_vx = actions[:,0] * torch.cos(theta)
        new_vy = actions[:,0] * torch.sin(theta)
        new_omega = actions[:,1]

        new_x = x + new_vx * self._dt
        new_y = y + new_vy * self._dt
        new_theta = theta + new_omega * self._dt

        new_states = torch.stack([new_x, new_y, new_theta, new_vx, new_vy, new_omega], dim=1)
        return new_states, actions
    
    def constant_velocity_model(self, states: torch.Tensor) -> torch.Tensor:
        x, y, theta, vx, vy, omega = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]

        # Tranform velocities to the global frame
        vx_global = vx * torch.cos(theta) - vy * torch.sin(theta)
        vy_global = vx * torch.sin(theta) + vy * torch.cos(theta)

        # Update position using constant velocity model
        new_x = x + vx_global * self._dt
        new_y = y + vy_global * self._dt
        new_theta = theta + omega * self._dt

        new_states = torch.stack([new_x, new_y, new_theta, vx, vy, omega], dim=1)
        return new_states
    
    def step(self, states: torch.Tensor, actions: torch.Tensor, t) -> torch.Tensor:
        new_robot_state, actions = self.bycicle_model(states[:,0:6], actions)
        new_obs_state = self.constant_velocity_model(states[:,6:12])
        # new_obs_state = states[:,6:12]

        new_states = torch.concat([new_robot_state, new_obs_state], dim=1)
        return new_states, actions
