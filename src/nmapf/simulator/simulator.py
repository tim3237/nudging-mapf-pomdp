import torch

from .rewards import RewardWithMessageCost
from .state import State

try:
    from tqdm import tqdm
except ModuleNotFoundError as _:
    # no process bar
    def tqdm(loop):
        return loop

DISCOUNT_GAMMA = 0.99  # State value estimation in get_simulation_history assumes this to be fixed to 0.99. See below.


def bootstrap_value_targets(rewards, final_state_value_estimates, discount_factor=DISCOUNT_GAMMA):
    """
    We calculate state values as sum of discounted rewards using estimates at the end of the horizon
    :param rewards:
    :param final_state_value_estimates:
    :param discount_factor:
    :return:
    """
    padded_rewards = torch.nn.functional.pad(
        torch.cat([rewards, final_state_value_estimates.unsqueeze(0)], dim=0),
        (0, 0, 0, rewards.shape[0]), value=0
    )
    discount_filter = torch.cat([
        torch.ones_like(rewards[:1, :1]), torch.ones_like(rewards[:, :1]) * discount_factor
    ], dim=0).cumprod(dim=0)
    bootstrapped_state_values = torch.nn.functional.conv1d(
        padded_rewards.permute(1, 0).unsqueeze(1), discount_filter.permute(1, 0).unsqueeze(1)).squeeze(1)
    return bootstrapped_state_values.permute(1, 0).contiguous()


class Simulator:
    def __init__(self, num_agents=1000, random_geometry=False, batch_size=1,
                 sim_device="cpu", cache_device="cpu", allocate_steps=200):
        self.sim_device = sim_device
        self.cache_device = cache_device
        self.state = State(
            num_agents,
            random_geometry=random_geometry,
            batch_size=batch_size,
            sim_device=sim_device,
            cache_device=cache_device,
            allocate_steps=allocate_steps)
        self.reward = RewardWithMessageCost()
        self.past = self.reward.observe_state(self.state)
        self.last_reward = 0

    def step(self, policy, cache_key=None):
        if cache_key is not None:
            policy_out = policy(self.state, last_step_rewards=self.last_reward, cache_key=cache_key)
        else:
            policy_out = policy(self.state, self.last_reward)
        if isinstance(policy_out, tuple):
            action, state_value = policy_out
        else:
            action, state_value = policy_out, None
        self.state.transition(action, state_value=state_value)
        now = self.reward.observe_state(self.state)
        self.last_reward = self.reward(self.past, now)
        self.state.new_rewards(self.last_reward)
        self.past = now
        return self.last_reward

    def get_simulation_history(self, policy=None):
        self.state.clear_allocation()
        if policy is not None:
            _, state_value = policy(self.state, self.last_reward)
            state_value = state_value.to(self.cache_device)
        else:
            # simple state value estimate: assume agent number to decrease linearly
            # variable naming assumes ConstantReward, although the only requirement is that 0 is the maximum reward
            # should be useful if the worst-case step reward is -#agents_left
            total_agents_left = -self.state.history.rewards[self.state.history.recorded_steps-1]
            last_step_rewards = self.state.history.rewards[self.state.history.recorded_steps-2]
            agents_left_since_last_step = - (last_step_rewards + total_agents_left)
            no_progress = agents_left_since_last_step <= 0
            if no_progress.all():
                state_value = 100 * -total_agents_left  # assumes DISCOUNT_GAMMA=0.99
            else:
                agents_left_since_last_step[no_progress] = total_agents_left[no_progress]
                max_steps = (total_agents_left // agents_left_since_last_step).ceil().int().max()
                estimated_future_numbers = torch.zeros(*total_agents_left.shape, max_steps,
                                                       device=agents_left_since_last_step.device)
                estimated_future_numbers[..., 0] = total_agents_left
                for i in range(max_steps - 1):
                    estimated_future_numbers[..., i + 1] = (
                            estimated_future_numbers[..., i] - agents_left_since_last_step).clamp(min=0)
                # for simulations without progress, we assume the minimum possible state value
                # assumes DISCOUNT_GAMMA=0.99
                estimated_future_numbers[..., 1][no_progress] = 100 * estimated_future_numbers[..., 0][no_progress]
                discount_influence = torch.ones_like(estimated_future_numbers[:1, 1:]) * DISCOUNT_GAMMA
                discount_influence[:, 0] = 1
                discount_influence = discount_influence.cumprod(dim=1)
                state_value = (-estimated_future_numbers[:, 1:] * discount_influence).sum(dim=-1)
        self.state.history.state_value = bootstrap_value_targets(self.state.history.rewards, state_value)
        return self.state.history


def rollout(policy, simulator: Simulator, num_steps: int):
    steps_done = 0
    for _ in tqdm(range(num_steps)):
        last_reward = simulator.step(policy)
        steps_done += 1
        if steps_done % 100 == 0:
            print("Simulated", steps_done, "steps. There are",
                  (simulator.state.agents.num_targets > 0).sum().item(), "agents left.")


