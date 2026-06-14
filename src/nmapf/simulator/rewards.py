"""
Reward functions may observe the current and previous state to return a scalar score.
The rewards defined here consist of the cost sum over all agents in the simulation ranging per-agent from 0 (already
despawned) to -1 (destination not yet reached, received a message, is stuck in a crowd, ...)
"""
import torch


class ConstantReward:
    """constant negative reward for each agent which is not yet at its destination"""
    @staticmethod
    def observe_state(state):
        return state.agents.num_targets

    def __call__(self, past, now):
        return -(now > 0).float().sum(dim=-1)


class RewardWithMessageCost:
    def __init__(self, message_cost=1e-3):
        self.message_cost = message_cost
        self.constant_cost = 1 - message_cost

    @staticmethod
    def observe_state(state):
        return state.agents.num_targets, state.messages[state.stored_steps - 1]

    def __call__(self, past, now):
        return -(self.constant_cost * (now[0] > 0).float().sum(dim=-1) +
                 self.message_cost * (now[1] != 0).float().sum(dim=-1))


class MessageCostAndBlockedPenalty:
    def __init__(self, message_cost=0.1, block_penalty=0.1):
        self.message_cost = message_cost
        self.block_penalty = block_penalty
        self.constant_cost = 1 - message_cost - block_penalty

    @staticmethod
    def observe_state(state):
        return (
            (state.agents.num_targets > 0).float().sum(dim=-1),
            (state.messages[state.stored_steps - 1] != 0).float().sum(dim=-1),
            (torch.conv2d(  # an agent is blocked on all four sides of which at most 2 are walls
                (0.9 * (state.grid_values[state.stored_steps] == 0) + (state.grid_values[state.stored_steps] == 2)
                 ).unsqueeze(1),
                torch.tensor(
                    [[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], device=state.grid_values.device)
            ).squeeze(1) >= 4.8).float().sum(dim=-1).sum(dim=-1)
            # for performance reasons, state.agents.is_blocked.sum(dim=-1) might be preferable
        )

    def __call__(self, past, now):
        return -(self.constant_cost * now[0] + self.message_cost * now[1] + self.block_penalty * now[2])
