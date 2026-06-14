import torch

from .grid import RoomGrid


def in_bounds(x, y, tensor):
    return 0 <= x < tensor.shape[0] and 0 <= y < tensor.shape[1]


class Navigation:
    """
    Keeps navigation information for some geometry grid.
    Computes the distance of each position to each of the defined target locations.
    """
    def __init__(self, geometry_grid: RoomGrid):
        sim_device = geometry_grid.geometry.device
        float_type = torch.float if sim_device == "cpu" else torch.float16
        inf_dist = geometry_grid.geometry.shape[1]*geometry_grid.geometry.shape[2]
        max_num_target_areas = max([len(t) for t in geometry_grid.target_areas])
        self.target_distances = [
            [torch.ones_like(geometry_grid.geometry[0]) * inf_dist for _ in range(max_num_target_areas)]
            for j in range(geometry_grid.geometry.shape[0])
        ]
        for j, target_distances in enumerate(self.target_distances):
            for a in geometry_grid.target_areas[j]:
                i = a-2
                target_distances[i][geometry_grid.geometry[j] == a] = 0
                changed = True
                while changed:
                    new_distances = target_distances[i].clone()
                    new_distances[1:] = torch.stack([new_distances[:-1]+1, new_distances[1:]], dim=0
                                                    ).min(dim=0).values * (geometry_grid[j, :-1] >= 0) + (
                                                    geometry_grid[j, :-1] < 0) * new_distances[1:]
                    new_distances[:-1] = torch.stack([new_distances[:-1], new_distances[1:]+1], dim=0
                                                     ).min(dim=0).values * (geometry_grid[j, 1:] >= 0) + (
                                                    geometry_grid[j, 1:] < 0) * new_distances[:-1]
                    new_distances[:, 1:] = torch.stack([new_distances[:, :-1]+1, new_distances[:, 1:]], dim=0
                                                       ).min(dim=0).values * (geometry_grid[j, :, :-1] >= 0) + (
                                                      geometry_grid[j, :, :-1] < 0) * new_distances[:, 1:]
                    new_distances[:, :-1] = torch.stack([new_distances[:, :-1], new_distances[:, 1:]+1], dim=0
                                                        ).min(dim=0).values * (geometry_grid[j, :, 1:] >= 0) + (
                                                        geometry_grid[j, :, 1:] < 0) * new_distances[:, :-1]
                    changed = target_distances[i].eq(new_distances).all().logical_not()
                    target_distances[i] = new_distances
                target_distances[i][geometry_grid.geometry[j] < 0] = inf_dist
        self.target_distances = torch.stack(
            [torch.stack(target_distances, dim=0) for target_distances in self.target_distances], dim=0)
        # target distance reduction for moving  west, east, not at all, north and south
        # we unsqueeze target_distances for a singleton channel dimension
        self.dir_values = torch.nn.functional.conv2d(self.target_distances.flatten(0, 1).unsqueeze(1).to(float_type),
                                                     torch.tensor(
                                                         [
                                                             [
                                                                 [0, -1, 0],
                                                                 [0, 1, 0],
                                                                 [0, 0, 0]
                                                             ],
                                                             [
                                                                 [0, 0, 0],
                                                                 [0, 1, 0],
                                                                 [0, -1, 0]
                                                             ],
                                                             [
                                                                 [0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]
                                                             ],
                                                             [
                                                                 [0, 0, 0],
                                                                 [-1, 1, 0],
                                                                 [0, 0, 0]
                                                             ],
                                                             [
                                                                 [0, 0, 0],
                                                                 [0, 1, -1],
                                                                 [0, 0, 0]
                                                             ]
                                                         ], dtype=float_type, device=sim_device).unsqueeze(1),
                                                     padding="same").permute(0, 2, 3, 1).unflatten(
            0, self.target_distances.shape[:2])
        self.dir_values[(self.dir_values < -1).logical_or(self.dir_values > 1)] = torch.finfo(float_type).min
        self.dir_values = self.dir_values.to(float_type)
        self.flat_dir_values = self.dir_values.flatten(0, 2)
        self.flat_distances_to_targets = self.target_distances.flatten()

    def as_flat_index(self, targets, positions):
        b_idx = torch.arange(self.dir_values.shape[0], device=targets.device).unsqueeze(1)
        if self.dir_values.shape[0] == 1 and positions.shape[0] > 1:
            b_idx = b_idx.expand(positions.shape[0], 1)  # manual broadcast index
        t_idx = (targets-2)
        x_idx = positions[..., 0].to(torch.int64)
        y_idx = positions[..., 1]
        flat_index = (((b_idx * self.dir_values.shape[1] + t_idx) * self.dir_values.shape[2] + x_idx) *
                      self.dir_values.shape[3] + y_idx)
        return flat_index

    def __getitem__(self, item):
        return self.target_distances[item[0]-2].__getitem__(item[1:])\
            if in_bounds(*item[1:], self.target_distances[item[0]-2]) else self.target_distances[item[0]-2].max()

    def get_distances_to_target(self, targets, positions):
        return self.flat_distances_to_targets[self.as_flat_index(targets, positions)]

    def get_dir_vals(self, targets, positions):
        """
        returns target distance reduction for moving  west, north, east and south
        :param targets:
        :param positions:
        :return:
        """
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.flat_dir_values.device)
        if len(targets.shape) == 0:  # single target only
            return self.flat_dir_values[self.as_flat_index(targets.unsqueeze(0), positions.unsqueeze(0))].squeeze(0)
        else:
            return self.flat_dir_values[self.as_flat_index(targets, positions)]