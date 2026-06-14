from .simulator.state import State
import torch


class NoOpPolicy:
    """
    Always produces the no-operation message 0
    """
    def __call__(self, state: State, last_step_rewards=None):
        return torch.zeros_like(state.agents.targets[:, :, 0])


class RareRandomPolicy:
    """
    Produces default messages with default_prob and random messages with probability 1-default_prob
    """
    def __init__(self, default_prob=0.9, max_target_areas=10, default_message=0):
        self.default_prob = default_prob
        self.num_non_null_messages = max_target_areas + 2
        self.default_message = default_message

    def __call__(self, state: State, last_step_rewards=None):
        messages = torch.full_like(state.agents.targets[..., 0], self.default_message)
        default_mask = torch.rand_like(state.agents.targets[..., 0], dtype=torch.float) > self.default_prob
        messages[default_mask] = torch.randint(
            low=0, high=self.num_non_null_messages+1, size=(default_mask.sum().item(),),
            dtype=messages.dtype, device=state.agents.positions.device)
        return messages

