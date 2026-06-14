"""
Simulation agents move towards their own destination. They are not path-planning agents optimising the global situation.
"""
import torch

DECISION_TEMPERATURE = 0.4


class Agents:
    def __init__(self, positions, targets):
        self.positions = positions
        self.targets = torch.stack([targets, torch.zeros_like(targets)], dim=-1)
        self.num_targets = torch.ones_like(targets)
        self.no_progress_steps = torch.zeros_like(targets)
        self.no_message_applied_steps = torch.zeros_like(targets)
        self.is_blocked = torch.zeros_like(targets, dtype=torch.bool)
        self.update_grid = None
        self.empty_grid_codes = None

    def pos_as_flat_idx(self):
        if self.update_grid is None:
            raise RuntimeError("Agents: Grid shape not yet known. Set it by calling setup_grid_values with an empty" +
                               "map of target shape.")
        return self.positions[:, :, 0].to(torch.int64) * self.update_grid.shape[2] + self.positions[:, :, 1]

    def setup_grid_values(self, grid_values):
        """
        Define dense grid for parallel updates without conflicts
        Each agent can only move left/right/up/down/in-place. Therefore, we can simultaneously update agents on a
        grid like
        [[+, o, o, o, o, +, o, o],
         [o, o, o, +, o, o, o, o],
         [o, +, o, o, o, o, +, o],
         [o, o, o, o, +, o, o, o],
         [o, o, +, o ,o ,o ,o ,+],
         [+, o, o, o, o, +, o, o]].
        """
        self.update_grid = ((
            torch.arange(grid_values.shape[2], device=grid_values.device).unsqueeze(0) +
            torch.arange(grid_values.shape[1], device=grid_values.device).unsqueeze(1) * 3
        ) % 5).eq(0).unsqueeze(0)
        self.empty_grid_codes = grid_values + 1
        # indicating agents with 1s within the input grid_value tensor
        grid_values.flatten(1).scatter_(1, self.pos_as_flat_idx(), 1)

    def update_positions(self, navigation, occupied, messages):
        """
        Simulates a movement step of all agents
        :param navigation: navigation object providing target-specific directions
        :param occupied: in/out argument. Shape (2, b, g_dim1, g_dim2). occupied[0] covers current position information
            mapped to the grid. occupied[1] is changed to hold the mapped positions after the update as model input.
        :param messages: messages for all agents given by a policy
        :return:
        """
        self.decide_for_intermediate_goal(messages)
        occupied[1] = occupied[0].clone()  # we keep occupied[0] unchanged and work in-place in occupied[1]
        targets_at_positions = torch.zeros_like(self.empty_grid_codes, dtype=torch.int64)
        targets_at_positions.flatten(1).scatter_(
            1, self.pos_as_flat_idx(),
            self.targets.gather(
                2, (self.num_targets.unsqueeze(-1).to(torch.int64)-1).clamp(min=0)
            ).squeeze(-1).to(torch.int64) - 2)
        update_map = navigation.dir_values.expand(
            targets_at_positions.shape[0], *navigation.dir_values.shape[1:]).gather(
            1, targets_at_positions.clamp(max=navigation.dir_values.shape[1] - 1).unsqueeze(1).unsqueeze(-1).expand(
                (targets_at_positions.shape[0], 1, *navigation.dir_values.shape[2:]))
        ).clone().squeeze(1)  # clamping is required since policies may return invalid messages for random rooms
        full_move_dirs = torch.zeros_like(occupied[1])
        is_blocked = torch.zeros_like(occupied[1], dtype=torch.bool)
        old_flat_index = self.pos_as_flat_idx()
        for grid_pos in torch.randperm(5).tolist():
            assert self.update_grid.shape[2] % 5 == 0, "The second grid dimension needs to be divisible by 5."
            update_mask = self.update_grid.roll(shifts=grid_pos, dims=2)  # roll in d2 direction assuming d2_size%5 == 0
            update_mask = update_mask.expand(update_map.shape[0], *update_mask.shape[1:])
            # mask out blocked directions
            blocked_positions = (occupied[1] == 2).logical_or(occupied[1] == 0)
            left_blocked_mask = torch.nn.functional.pad(
                blocked_positions[:, :-1], (0, 0, 1, 0), mode='constant', value=True)
            right_blocked_mask = torch.nn.functional.pad(
                blocked_positions[:, 1:], (0, 0, 0, 1), mode='constant', value=True)
            top_blocked_mask = torch.nn.functional.pad(
                blocked_positions[:, :, :-1], (1, 0), mode='constant', value=True)
            bot_blocked_mask = torch.nn.functional.pad(
                blocked_positions[:, :, 1:], (0, 1), mode='constant', value=True)
            blocked_mask = torch.stack(
                [left_blocked_mask, right_blocked_mask, torch.zeros_like(left_blocked_mask), top_blocked_mask,
                 bot_blocked_mask], dim=-1).logical_and(update_mask.unsqueeze(-1))
            update_map[blocked_mask] = torch.finfo(update_map.dtype).min

            # determine which active positions do not allow any movement
            is_blocked[update_mask] = (
                    update_map[update_mask] == torch.finfo(update_map.dtype).min
            )[:, [0, 1, 3, 4]].all(dim=-1)

            # choose movement direction based on predicted scores
            if DECISION_TEMPERATURE == 0:
                update_directions = update_map.argmax(dim=-1)
            else:
                shape = update_map.shape[:-1]
                update_directions = (update_map.flatten(0, -2)/DECISION_TEMPERATURE).softmax(dim=-1).multinomial(1)
                update_directions = update_directions.reshape(shape)
            update_directions[update_mask.logical_not()] = 2  # set directions of not active agents to "don't move"
            update_directions[0, 0] = 2
            full_move_dirs[update_mask] = update_directions[update_mask].to(full_move_dirs.dtype)

            # identify which agents move onto a new tile
            # use old occupied codes here to not double-move agents
            movement_mask = (occupied[0] == 2).logical_and(update_directions != 2)

            # clear moved agent tiles
            occupied[1][movement_mask] = self.empty_grid_codes[movement_mask].clone()
            # place moved agents to their new tiles
            occupied[1][:, :-1][movement_mask.logical_and(update_directions == 0)[:, 1:]] = 2
            occupied[1][:, 1:][movement_mask.logical_and(update_directions == 1)[:, :-1]] = 2
            occupied[1][:, :, :-1][movement_mask.logical_and(update_directions == 3)[:, :, 1:]] = 2
            occupied[1][:, :, 1:][movement_mask.logical_and(update_directions == 4)[:, :, :-1]] = 2

        # determine which agents made progress in this step
        is_progress = torch.gather(
            update_map.flatten(1, 2).gather(2, full_move_dirs.flatten(1).unsqueeze(1).long()).squeeze(1),
            1, self.pos_as_flat_idx()
        ) > 0
        self.no_progress_steps += 1
        self.no_progress_steps[is_progress] = 0

        # update agent's positions
        agent_movements = full_move_dirs.flatten(1).gather(1, self.pos_as_flat_idx())
        self.positions[
            agent_movements == 0
            ] = self.positions[agent_movements == 0] - torch.tensor(
            [[[1, 0]]], dtype=self.positions.dtype, device=self.positions.device)
        self.positions[
            agent_movements == 1
            ] = self.positions[agent_movements == 1] + torch.tensor(
            [[[1, 0]]], dtype=self.positions.dtype, device=self.positions.device)
        self.positions[
            agent_movements == 3
            ] = self.positions[agent_movements == 3] - torch.tensor(
            [[[0, 1]]], dtype=self.positions.dtype, device=self.positions.device)
        self.positions[
            agent_movements == 4
            ] = self.positions[agent_movements == 4] + torch.tensor(
            [[[0, 1]]], dtype=self.positions.dtype, device=self.positions.device)

        # check if agents reached a goal
        agent_tile_values = self.empty_grid_codes.flatten(1).gather(1, self.pos_as_flat_idx())
        # intermediate targets
        self.num_targets[self.targets[:, :, 1] == agent_tile_values-1] = 1
        # main target
        finished_agents = self.targets[:, :, 0] == agent_tile_values-1
        # Despawning does not happen instantly. Allowing other agents to move onto not yet freed target area tiles might
        # become relevant in more general terminal target area layouts.
        occupied[1].flatten(1).scatter_(1, self.pos_as_flat_idx(), torch.where(
            finished_agents, agent_tile_values, occupied[1].flatten(1).gather(1, self.pos_as_flat_idx())))
        self.positions[finished_agents] = 0
        self.num_targets[finished_agents] = 0

        # Store which agents could not move at all. Could be used in reward functions.
        self.is_blocked = is_blocked.flatten(1).gather(1, old_flat_index)

    def dir_vals(self, navigation):
        return navigation.get_dir_vals(self.targets, self.positions)

    def decide_for_intermediate_goal(self, messages):
        # 0: no message
        # 1: drop sub-goal
        apply_message_mask = (messages > 0).logical_and(self.num_targets > 0)
        # and torch.rand_like(messages.float()) < 1/(self.no_progress_steps+1)

        # drop target
        self.num_targets[
            apply_message_mask.logical_and((messages == 1).logical_or(self.targets[..., 0] == messages))] = 1

        # set new targets
        set_secondary_target_mask = apply_message_mask.logical_and(messages > 1)
        self.targets[..., 1][set_secondary_target_mask] = messages[set_secondary_target_mask]
        self.num_targets[set_secondary_target_mask] = 2

        # Keep track of time since last applied message. Could be used for agent attention simulation.
        self.no_message_applied_steps[apply_message_mask] = 0
        self.no_message_applied_steps[apply_message_mask.logical_not()] += 1



