import torch

from .random_room_layout import setup_random_room


class RoomGrid:
    """
    Keeps information of geometry like walls and different target areas
    """
    def __init__(self, random_geometry=True, batch_size=1, sim_device="cpu"):
        if random_geometry:
            self.geometry = torch.stack([
                setup_random_room().to(sim_device) for _ in range(batch_size)], dim=0)
        else:
            self.geometry = torch.tensor([
                [-1]*47 + [2]*6 + [-1]*47,
                *[[-1] + [0]*98 + [-1]]*30,
                *[[-1] + [4]*5 + [-1]*41 + [5]*6 + [-1]*47] * 5,
                *[[-1] + [0]*98 + [-1]]*30,
                *[[-1] + [6]*5 + [-1]*61 + [7]*6 + [-1]*27] * 5,
                *[[-1] + [0]*98 + [-1]]*30,
                [-1]*47 + [3]*6 + [-1]*47
            ], device=sim_device).unsqueeze(0)  # broadcastable over batch dimension
        self.target_areas = [  # 1 is reserved for "there is an agent"
            self.geometry[i, self.geometry[i] > 1].unique().tolist() for i in range(self.geometry.shape[0])]
        self.terminal_areas = [
            torch.tensor([
                a for a in (self.geometry[i] * torch.nn.functional.pad(
                    torch.zeros_like(self.geometry[i, :-2, :-2]), (1, 1, 1, 1), value=1)
                        ).unique().tolist() if a > 1], dtype=torch.int8, device=sim_device)
            for i in range(self.geometry.shape[0])]

    def __getitem__(self, item):
        return self.geometry.__getitem__(item)

