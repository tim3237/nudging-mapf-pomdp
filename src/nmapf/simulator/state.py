import torch

from .agents import Agents
from .grid import RoomGrid
from .navigation import Navigation


class HistoryCollector:
    def __init__(self,
                 geometry,
                 agent_targets,
                 tile_data=None,
                 position_data=None,
                 messages=None,
                 state_value=None,
                 num_goals=None,
                 rewards=None,
                 device="cpu"):
        self.device = device
        if rewards is None:  # setup blank history to be filled using batched simulation
            batch_size, num_agents = agent_targets.shape
            self.agent_targets = agent_targets.to( self.device) - 2  # shift target labels from 2... to 0...
            self.geometry = (geometry + 1).expand(batch_size, *geometry.shape[1:]).unsqueeze(0)
            # shift {-1: wall, 0: free, 1: agent, 2: target area 1, 3: target area 2, ...} to
            #       { 0: wall, 1: free, 2: agent, 3: target area 1, ...}

            self.tile_data = torch.ones((0, *self.geometry.shape[1:]), dtype=torch.uint8, device=self.device)

            self.position_data = torch.zeros(
                (0, batch_size, num_agents, 2), dtype=torch.uint8, device=self.device)
            # assuming 0,0 is an invalid position for active agents

            self.messages = -torch.ones((0, batch_size, num_agents), dtype=torch.int32, device=self.device)
            self.state_value = None  # will be calculated externally after self.rewards are completely filled
            self.num_goals = torch.ones((0, batch_size, num_agents), dtype=torch.uint8, device=self.device)

            self.rewards = torch.zeros((0, batch_size), dtype=torch.float, device=self.device)
            self.recorded_steps = 0
        else:  # store single simulation history for easily accessible pickle
            self.agent_targets = agent_targets
            self.geometry = geometry
            self.tile_data = tile_data
            self.position_data = position_data
            self.messages = messages
            self.state_value = state_value
            self.num_goals = num_goals
            self.rewards = rewards
            self.recorded_steps = rewards.shape[0]

    def __call__(self, grid_values, positions, messages, num_goals, rewards):
        self.tile_data = torch.cat([self.tile_data, grid_values.to(self.device)], dim=0)
        self.position_data = torch.cat([self.position_data, positions.to(self.device)], dim=0)
        self.messages = torch.cat([self.messages, messages.to(self.device)], dim=0)
        self.num_goals = torch.cat([self.num_goals, num_goals.to(self.device)], dim=0)
        self.rewards = torch.cat([self.rewards, rewards.to(self.device)], dim=0)
        self.recorded_steps += rewards.shape[0]

    def get_data(self):
        return self.tile_data[:self.recorded_steps+1], self.position_data[:self.recorded_steps+1]

    def get_agent_position_list(self, include_num_goals=True, select_from_batch=0):
        non_null_pos = (self.position_data[:, select_from_batch] != torch.zeros_like(self.position_data[:1, 0, :1])).any(dim=-1)
        if include_num_goals:
            all_data = torch.cat(
                [self.position_data[:, select_from_batch], self.num_goals[:, select_from_batch].unsqueeze(-1)], dim=-1)
        else:
            all_data = self.position_data[:, select_from_batch]
        return [all_data[i][non_null_pos[i]].tolist() for i in range(all_data.shape[0])]

    def decompose_batch(self):
        return [self.single_sample_history(i) for i in range(self.agent_targets.shape[1])]

    def single_sample_history(self, i):
        return HistoryCollector(
            self.geometry[:, i].clone(),
            self.agent_targets[:, i].clone(),
            tile_data=self.tile_data[:, i].clone(),
            position_data=self.position_data[:, i].clone(),
            messages=self.messages[:, i].clone(),
            state_value=self.state_value[:, i].clone(),
            num_goals=self.num_goals[:, i].clone(),
            rewards=self.rewards[:, i].clone())


class State:
    def __init__(self,
                 num_agents: int,
                 allocate_steps=200,
                 do_save_history=True,
                 random_geometry=True,
                 batch_size=1,
                 sim_device="cpu",
                 cache_device="cpu"):
        self.geometry = RoomGrid(random_geometry=random_geometry, sim_device=sim_device)
        self.navigation = Navigation(self.geometry)

        # spawn num_agents agents at random positions
        _, flattened_positions = (
                torch.rand(
                    (batch_size, self.geometry.geometry[0].numel())
                    , dtype=torch.float, device=sim_device
                           ) * (self.geometry.geometry.flatten(1) == 0).float()
        ).topk(num_agents, dim=-1)
        positions = torch.stack([flattened_positions // self.geometry.geometry.shape[2],
                                 flattened_positions % self.geometry.geometry.shape[2]], dim=-1)
        # setup random targets
        if len(self.geometry.terminal_areas) == 1 and batch_size > 1:
            tas = [self.geometry.terminal_areas[0]] * batch_size  # broadcast target areas
        else:
            tas = self.geometry.terminal_areas
        targets = torch.stack(
            [tas[b][torch.randint(0, len(tas[b]), (num_agents,), device=sim_device)]
             for b in range(batch_size)], dim=0)

        self.agents = Agents(positions, targets)
        self.messages = torch.zeros((allocate_steps, *positions[..., 0].shape), dtype=torch.int8, device=sim_device)
        self.positions = torch.zeros((allocate_steps+1, *positions.shape), dtype=torch.uint8, device=sim_device)
        self.num_goals = torch.zeros((allocate_steps+1, *positions[..., 0].shape), dtype=torch.int8, device=sim_device)
        self.rewards = torch.zeros((allocate_steps, *positions[..., 0, 0].shape))
        self.positions[0] = self.agents.positions.clone()
        self.num_goals[0] = self.agents.num_targets.clone()
        self.grid_values = self.geometry.geometry.unsqueeze(0).expand(
            (allocate_steps+1, self.positions.shape[1], *self.geometry.geometry.shape[1:])).clone()
        self.agents.setup_grid_values(self.grid_values[0])  # updates timestep 0 positions in-place
        self.grid_values = self.grid_values.contiguous()
        self.grid_values += 1
        # shift {-1: wall, 0: free, 1: agent, 2: target area 1, 3: target area 2, ...} to
        #       { 0: wall, 1: free, 2: agent, 3: target area 1, ...}
        self.history = HistoryCollector(self.geometry.geometry, agent_targets=targets.clone(), device=cache_device)
        self.allocate_steps = allocate_steps
        self.do_save_history = do_save_history
        self.stored_steps = 0

    def new_rewards(self, rewards):
        self.rewards[self.stored_steps-1] = rewards

    def get_history_data(self):
        if self.history.position_data.shape[0] == 0:
            return self.grid_values[:self.stored_steps+1], self.positions[:self.stored_steps+1]
        else:
            history_tiles, history_positions = self.history.get_data()
            return torch.cat(
                [history_tiles, self.grid_values[1:self.stored_steps+1]], dim=0
            ), torch.cat([history_positions, self.positions[1:self.stored_steps+1]], dim=0)

    def get_prediction_data(self):
        return (self.history.recorded_steps + self.stored_steps,
                self.grid_values[self.stored_steps],
                self.agents.positions)

    def clear_allocation(self):
        """
        Writes working data to history. Should always be called before history is used.
        """
        if self.history.position_data.shape[0] == 0:
            init_position_offset = 0
        else:
            init_position_offset = 1
        if self.do_save_history:
            self.history(
                self.grid_values[init_position_offset:self.stored_steps+1],
                self.positions[init_position_offset:self.stored_steps+1],
                self.messages[:self.stored_steps],
                self.num_goals[init_position_offset:self.stored_steps+1],
                self.rewards[:self.stored_steps])

        # clear memory positions
        last_grid_values = self.grid_values[self.stored_steps]
        self.grid_values = torch.zeros_like(self.grid_values)
        self.grid_values[0] = last_grid_values
        self.positions = torch.zeros_like(self.positions)
        self.messages = torch.zeros_like(self.messages)
        self.num_goals = torch.zeros_like(self.num_goals)
        self.rewards = torch.zeros_like(self.rewards)

        # reset number of stored time steps
        self.stored_steps = 0

    def transition(self, action, state_value=None):
        # check if next batch should be stored in the history object
        if self.stored_steps == self.allocate_steps:
            self.clear_allocation()

        self.messages[self.stored_steps] = action

        self.grid_values[self.stored_steps + 1] = self.grid_values[self.stored_steps].clone()
        self.agents.update_positions(
            self.navigation,
            self.grid_values[self.stored_steps:self.stored_steps+2],
            action
        )
        # store agent data
        self.positions[self.stored_steps+1] = self.agents.positions.clone()
        self.num_goals[self.stored_steps+1] = self.agents.num_targets.clone()

        self.stored_steps += 1

