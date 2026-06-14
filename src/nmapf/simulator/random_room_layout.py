import torch


def border_pos_to_2d_pos(flat_pos):
    """
    given a position in the flattened border tile tensor, this function returns respective x and y positions in the
    final grid.
    """
    flat_pos = flat_pos.item() if isinstance(flat_pos, torch.Tensor) else flat_pos
    if 0 <= flat_pos < 100:
        y = flat_pos
        x = 0
    elif 100 <= flat_pos < 200:
        y = 99
        x = flat_pos - 99
    elif 200 <= flat_pos < 300:
        y = 299 - flat_pos
        x = 101
    else:  # if flat_pos >= 300:
        y = 0
        x = 400 - flat_pos
    return x, y


def distance_step(distance_matrix, wall_dist=-1):
    is_wall = distance_matrix == wall_dist
    distance_matrix[:-1] = torch.stack(
        [distance_matrix[:-1], distance_matrix[1:] + 1], dim=0).min(dim=0).values
    distance_matrix[1:] = torch.stack(
        [distance_matrix[1:], distance_matrix[:-1] + 1], dim=0).min(dim=0).values
    distance_matrix[:, :-1] = torch.stack(
        [distance_matrix[:, :-1], distance_matrix[:, 1:] + 1], dim=0).min(dim=0).values
    distance_matrix[:, 1:] = torch.stack(
        [distance_matrix[:, 1:], distance_matrix[:, :-1] + 1], dim=0).min(dim=0).values
    distance_matrix[is_wall] = wall_dist


def wall_tiles_to_next_opening(room_grid, start_position, end_position=None, num_wall_tiles=None):
    """
    counts wall grid tiles on the edge of a sub-area starting at start_position, moving counter-clock-wise
    ([-> x- -> y+ -> x+ -> y-...]) to
    end_positions

    Note: Expects walls to extend at least 2 grid positions in new directions
    """
    x, y = start_position
    wall_positions = 0
    while end_position is not None and (x != end_position[0] or y != end_position[1]) or\
            num_wall_tiles is not None and wall_positions < num_wall_tiles:
        old_x, old_y = x, y
        if y > 0 and room_grid[x, y-1] == 0:  # wall is in y+ direction
            if x < room_grid.shape[0]-1 and room_grid[x+1, y] != 0 and room_grid[x+1, y-1] == 0:
                x = x+1  # move along straight wall
            elif x < room_grid.shape[0]-1 and room_grid[x+1, y] != 0:
                x = x+1  # move around corner to y-
                y = y-1
            else:
                y = y+1  # move around corner to y+
        elif x < room_grid.shape[0]-1 and room_grid[x+1, y] == 0:  # wall is in x- direction:
            if y < room_grid.shape[1]-1 and room_grid[x, y+1] != 0 and room_grid[x+1, y+1] == 0:
                y = y+1  # move along straight wall
            elif y < room_grid.shape[1]-1 and room_grid[x, y+1] != 0:
                x = x+1  # move around corner to x+
                y = y+1
            else:
                x = x-1  # move around corner to x-
        elif y < room_grid.shape[1]-1 and room_grid[x, y+1] == 0: # wall is in y- direction
            if x > 0 and room_grid[x-1, y] != 0 and room_grid[x-1, y+1] == 0:
                x = x-1  # move along straight wall
            elif x > 0 and room_grid[x-1, y] != 0:
                x = x-1  # move around corner to y+
                y = y+1
            else:
                y = y-1  # move around corner to y-
        elif x > 0 and room_grid[x-1, y] == 0:  # wall is in x+ direction:
            if y > 0 and room_grid[x, y-1] != 0 and room_grid[x-1, y-1] == 0:
                y = y-1  # move along straight wall
            elif y > 0 and room_grid[x, y-1] != 0:
                x = x-1  # move around corner to x+
                y = y-1
            else:
                x = x+1  # move around corner to x-

        if room_grid[x, y] == -1:
            wall_positions += 1
        assert not (old_x == x and old_y == y), ("invalid move while counting wall tiles, " +
                                                 "check the following grid for validity:\n" +
                                                 "\n".join([str(l) for l in room_grid.tolist()]))
    return wall_positions if end_position is not None else (x, y)


def get_dir_vals(vals, x, y):
    n = vals[x, max(y-1, 0)]
    e = vals[min(x+1, vals.shape[0]-1), y]
    s = vals[x, min(y+1, vals.shape[1]-1)]
    w = vals[max(x-1, 0), y]
    return n, e, s, w


def update_pos(x, y, direction):
    if direction == 0:
        return x, y-1
    elif direction == 1:
        return x+1, y
    elif direction == 2:
        return x, y+1
    else:
        return x-1, y


def draw_thin_wall(start, end, distance_from_anything):
    """
    Draws a 1-tile thick wall connecting start and end positions. Except for at start and end, the wall must be at least
    5 tiles away from anything.
    :param start:
    :param end:
    :param distance_from_anything:
    :return:
    """
    # setting up path navigation for building new wall
    max_dist = 9e9
    start_dist = 5e5
    min_dir_steps = 10
    wall_navigation = start_dist * torch.ones_like(distance_from_anything)
    wall_navigation[distance_from_anything == 0] = max_dist
    wall_navigation[start[0], start[1]] = start_dist
    wall_navigation[end[0], end[1]] = 0
    last_avg_distance = wall_navigation.mean()
    has_changed = True
    while has_changed:
        distance_step(wall_navigation)
        avg_distance = wall_navigation.mean()
        has_changed = avg_distance != last_avg_distance
        last_avg_distance = avg_distance

    wall_tiles = torch.zeros_like(wall_navigation)
    x, y = start
    # moving from start directly to some position, where anything else is sufficiently far away
    dirv = get_dir_vals(distance_from_anything, x, y)
    move_dir = dirv.index(max(dirv))
    x, y = update_pos(x, y, move_dir)
    move_dir_steps = 0
    while distance_from_anything[x, y] < 10:
        wall_tiles[x, y] = 1
        dirv = get_dir_vals(distance_from_anything, x, y)
        if max(dirv) == distance_from_anything[x, y]:  # we would need to move diagonally
            return None
        if dirv[move_dir] != max(dirv):
            move_dir = dirv.index(max(dirv))
            move_dir_steps = 0
        x, y = update_pos(x, y, move_dir)
        move_dir_steps += 1

    # moving further to reach target
    abort = False
    while (x != end[0] or y != end[1]) and not abort:
        if wall_tiles[x, y] == 1:
            abort = True  # this position has already been visited
        wall_tiles[x, y] = 1
        dir_dta = get_dir_vals(distance_from_anything, x, y)
        dirv = get_dir_vals(wall_navigation, x, y)

        prev_move_dir = move_dir
        if dirv[move_dir] != min(dirv) or dir_dta[move_dir] < 10:
            best_nav_dir = dirv.index(min(dirv)) if move_dir_steps >= min_dir_steps else move_dir
            if dir_dta[best_nav_dir] >= 10 or dir_dta[best_nav_dir] == abs(x - end[0]) + abs(y - end[1]) - 1:
                move_dir = best_nav_dir
            elif dir_dta[move_dir] < 10:
                abort = True

        if prev_move_dir == move_dir:
            move_dir_steps += 1
        else:
            move_dir_steps = 0

        if move_dir == 0:
            y -= 1
        elif move_dir == 1:
            x += 1
        elif move_dir == 2:
            y += 1
        else:
            x -= 1
    if abort:
        return None
    else:
        return wall_tiles


def insert_side_wall(start_end_wall, room_grid, x, y, wall_start_dir):
    """
    Inserts wall positions taken from room_grid into start_end_wall to both sides of position (x, y).
    Since we want to find out whether there are limits for the width of a wall starting from (x, y) in wall_start_dir,
    we count the number of available straight wall tiles to either side. This way, we can ensure that the new wall will
    start and end at existing wall without extending over edges.
    :param start_end_wall:
    :param room_grid:
    :param x:
    :param y:
    :param wall_start_dir:
    :return:
    """
    max_limit = 20
    if wall_start_dir == 0 or wall_start_dir == 2:  # look for wall in east-west direction
        room_y = y-1 if wall_start_dir == 0 else y+1
        east_wall_len = 0
        i = 1
        while room_grid[x+i, y] == -1 and room_grid[x+i, room_y] == 0:
            east_wall_len += 1
            start_end_wall[x+i, y] = 1
            i += 1
        west_wall_len = 0
        i = 1
        while room_grid[x-i, y] == -1 and room_grid[x-i, room_y] == 0:
            west_wall_len += 1
            start_end_wall[x-i, y] = 1
            i += 1
        north_wall_len, south_wall_len = max_limit, max_limit
    else:  # look for wall in north-south direction
        room_x = x+1 if wall_start_dir == 1 else x-1
        north_wall_len = 0
        i = 1
        while room_grid[x, y-i] == -1 and room_grid[room_x, y-i] == 0:
            north_wall_len += 1
            start_end_wall[x, y-i] = 1
            i += 1
        south_wall_len = 0
        i = 1
        while room_grid[x, y+i] == -1 and room_grid[room_x, y+i] == 0:
            south_wall_len += 1
            start_end_wall[x, y+i] = 1
            i += 1
        east_wall_len, west_wall_len = max_limit, max_limit
    return north_wall_len, east_wall_len, south_wall_len, west_wall_len


def random_width(thin_wall, start, end, distance_from_anything, room_grid):
    if thin_wall is None:
        return None
    # find areas where the new wall should touch other walls
    start_end_wall = torch.zeros_like(thin_wall)
    wall_start_vals = get_dir_vals(thin_wall, *start)
    wall_start_dir = wall_start_vals.index(max(wall_start_vals))
    # setup wall bounds to cut off overlapping parts from the widened wall over touched walls
    if wall_start_dir == 0:
        start_end_wall[max(start[0]-10, 0):min(start[0]+11,start_end_wall.shape[0]),
                       start[1]:min(start[1]+20, start_end_wall.shape[1])] = 1
    elif wall_start_dir == 2:
        start_end_wall[max(start[0]-10, 0):min(start[0]+11, start_end_wall.shape[0]),
                       max(start[1]-20, 0):start[1]+1] = 1
    elif wall_start_dir == 1:
        start_end_wall[max(start[0]-20, 0):start[0]+1,
                       max(start[1]-10, 0):min(start[1]+11, start_end_wall.shape[1])] = 1
    else:
        start_end_wall[start[0]:min(start[0]+20, start_end_wall.shape[0]),
                       max(start[1]-10, 0):min(start[1]+11, start_end_wall.shape[1])] = 1
    start_direction_limits = insert_side_wall(start_end_wall, room_grid, *start, wall_start_dir)
    wall_end_vals = get_dir_vals(thin_wall, *end)
    wall_end_dir = wall_end_vals.index(max(wall_end_vals))
    if wall_end_dir == 0:
        start_end_wall[max(end[0]-10, 0):min(end[0]+11, start_end_wall.shape[0]),
                       end[1]:min(end[1]+20, start_end_wall.shape[1])] = 1
    elif wall_end_dir == 2:
        start_end_wall[max(end[0]-10, 0):min(end[0]+11, start_end_wall.shape[0]), max(end[1]-20, 0):end[1]+1] = 1
    elif wall_end_dir == 1:
        start_end_wall[max(end[0]-20, 0):end[0]+1, max(end[1]-10, 0):min(end[1]+11, start_end_wall.shape[1])] = 1
    else:
        start_end_wall[end[0]:min(end[0]+20, start_end_wall.shape[0]),
                       max(end[1]-10, 0):min(end[1]+11, start_end_wall.shape[1])] = 1
    end_direction_limits = insert_side_wall(start_end_wall, room_grid, *end, wall_end_dir)
    direction_limits = [min(a, b) for a, b in zip(start_direction_limits, end_direction_limits)]
    # find direction limits due to other walls
    # north
    work_wall = thin_wall.clone()
    for i in range(direction_limits[0]):
        work_wall = torch.nn.functional.pad(work_wall[:, 1:], (0, 1))
        work_wall[work_wall == start_end_wall] = 0  # erase ends moved into start/end walls
        if ((distance_from_anything[work_wall == 1] == 0).any() or
                (distance_from_anything[work_wall == 1] < 10).sum() > 20):
            direction_limits[0] = i
            break
    # east
    work_wall = thin_wall.clone()
    for i in range(direction_limits[1]):
        work_wall = torch.nn.functional.pad(work_wall[:-1], (0, 0, 1, 0))
        work_wall[work_wall == start_end_wall] = 0  # erase ends moved into start/end walls
        if ((distance_from_anything[work_wall == 1] == 0).any() or
                (distance_from_anything[work_wall == 1] < 10).sum() > 20):
            direction_limits[1] = i
            break
    # south
    work_wall = thin_wall.clone()
    for i in range(direction_limits[2]):
        work_wall = torch.nn.functional.pad(work_wall[:, :-1], (1, 0))
        work_wall[work_wall == start_end_wall] = 0  # erase ends moved into start/end walls
        if ((distance_from_anything[work_wall == 1] == 0).any() or
                (distance_from_anything[work_wall == 1] < 10).sum() > 20):
            direction_limits[2] = i
            break
    # west
    work_wall = thin_wall.clone()
    for i in range(direction_limits[3]):
        work_wall = torch.nn.functional.pad(work_wall[1:], (0, 0, 0, 1))
        work_wall[work_wall == start_end_wall] = 0  # erase ends moved into start/end walls
        if ((distance_from_anything[work_wall == 1] == 0).any() or
                (distance_from_anything[work_wall == 1] < 10).sum() > 20):
            direction_limits[3] = i
            break

    # decide for one direction each out of (north, south) and (east, west) to extend the wall by picking the max-min of
    # the bounds of taken directions
    north = direction_limits[0] > direction_limits[2]
    east = direction_limits[1] > direction_limits[3]
    min_limit = min(direction_limits[0 if north else 2], direction_limits[1 if east else 3])
    if min_limit < 2:
        return None  # there is no space make the wall wider
    # set wall width and widen thin_wall
    wall_width = torch.randint(low=2, high=min(10, min_limit+1), size=(1,)).item()
    wall = thin_wall.clone()
    for i in range(1, wall_width):
        if north:
            wall[:, :-i] = wall[:, :-i] + thin_wall[:, i:]*5
            if east:
                wall[:-i] = wall[:-i] + thin_wall[i:]*3
                wall[:-i, :-i] = wall[:-i, :-i] + thin_wall[i:, i:]*7
            else:
                wall[i:] = wall[i:] + thin_wall[:-i]*3
                wall[i:, :-i] = wall[i:, :-i] + thin_wall[:-i, i:]*7
        else:
            wall[:, i:] = wall[:, i:] + thin_wall[:, :-i]*5
            if east:
                wall[:-i] = wall[:-i] + thin_wall[i:]*3
                wall[:-i, i:] = wall[:-i, i:] + thin_wall[i:, :-i]*7
            else:
                wall[i:] = wall[i:] + thin_wall[:-i]*3
                wall[i:, i:] = wall[i:, i:] + thin_wall[:-i, :-i]*7
        wall[start_end_wall == 1] = 0
    wall = wall.clamp(max=1)
    return wall


def insert_openings(wall, start, opening_area_ids):
    if wall is None:
        return None
    # find straight passages which could fit at least one opening
    start_dir_v = get_dir_vals(wall, *start)
    wall_start_dir = start_dir_v.index(max(start_dir_v))
    if wall_start_dir == 0 or wall_start_dir == 2:  # north or south
        left_x = start[0]
        left_y = start[1] - 1 if wall_start_dir == 0 else start[1] + 1
        for i in range(min(20, start[0])):
            left_x = left_x - 1 if wall_start_dir == 0 else left_x + 1
            if wall[left_x, left_y] == 0:
                break
        right_x = start[0]
        right_y = left_y
        for i in range(min(20, wall.shape[0] - start[0])):
            right_x = right_x + 1 if wall_start_dir == 0 else right_x - 1
            if wall[right_x, right_y] == 0:
                break
    else:  # east or west
        left_y = start[1]
        left_x = start[0] + 1 if wall_start_dir == 1 else start[0] - 1
        for i in range(min(20, start[1])):
            left_y = left_y - 1 if wall_start_dir == 1 else left_y + 1
            if wall[left_x, left_y] == 0:
                break
        right_y = start[1]
        right_x = left_x
        for i in range(min(20, wall.shape[1] - start[1])):
            right_y = right_y + 1 if wall_start_dir == 1 else right_y - 1
            if wall[right_x, right_y] == 0:
                break
    wall_blocks = []  # rectangular parts of the wall without corners
    wall_dir = wall_start_dir
    while wall_dir >= 0:
        start_left = (left_x, left_y)
        start_right = (right_x, right_y)
        if wall_dir == 0:  # north
            while (wall[left_x + 1, left_y] == 1 and wall[right_x - 1, right_y] == 1 and
                   wall[left_x, left_y] == 0 and wall[right_x, right_y] == 0):
                left_y, right_y = left_y - 1, right_y - 1
            wall_blocks.append((wall_dir, start_left[1] - left_y - 1, (start_left, start_right)))
            if wall[left_x, left_y] == 1:  # corner to west
                right_x = left_x
                while wall[right_x, right_y] == 1:
                    right_y = right_y - 1
                left_y = left_y + 1
                wall_dir = 3
            elif wall[right_x, right_y] == 1:  # corner to east
                left_x = right_x
                while wall[left_x, left_y] == 1:
                    left_y = left_y - 1
                right_y = right_y + 1
                wall_dir = 1
            else:  # wall end
                wall_dir = -1
        elif wall_dir == 1:  # east
            while (wall[left_x, left_y + 1] == 1 and wall[right_x, right_y - 1] == 1 and
                   wall[left_x, left_y] == 0 and wall[right_x, right_y] == 0):
                left_x, right_x = left_x + 1, right_x + 1
            wall_blocks.append((wall_dir,  left_x - start_left[0] - 1, (start_left, start_right)))
            if wall[left_x, left_y] == 1:  # corner to north
                right_y = left_y
                while wall[right_x, right_y] == 1:
                    right_x = right_x + 1
                left_x = left_x - 1
                wall_dir = 0
            elif wall[right_x, right_y] == 1:  # corner to south
                left_y = right_y
                while wall[left_x, left_y] == 1:
                    left_x = left_x + 1
                right_x = right_x - 1
                wall_dir = 2
            else:  # wall end
                wall_dir = -1
        elif wall_dir == 2:  # south
            while (wall[left_x - 1, left_y] == 1 and wall[right_x + 1, right_y] == 1 and
                   wall[left_x, left_y] == 0 and wall[right_x, right_y] == 0):
                left_y, right_y = left_y + 1, right_y + 1
            wall_blocks.append((wall_dir,  left_y - start_left[1] - 1, (start_left, start_right)))
            if wall[left_x, left_y] == 1:  # corner to east
                right_x = left_x
                while wall[right_x, right_y] == 1:
                    right_y = right_y + 1
                left_y = left_y - 1
                wall_dir = 1
            elif wall[right_x, right_y] == 1:  # corner to east
                left_x = right_x
                while wall[left_x, left_y] == 1:
                    left_y = left_y + 1
                right_y = right_y - 1
                wall_dir = 3
            else:  # wall end
                wall_dir = -1
        elif wall_dir == 3:  # west
            while (wall[left_x, left_y - 1] == 1 and wall[right_x, right_y + 1] == 1 and
                   wall[left_x, left_y] == 0 and wall[right_x, right_y] == 0):
                left_x, right_x = left_x - 1, right_x - 1
            wall_blocks.append((wall_dir, start_left[0] - left_x - 1, (start_left, start_right)))
            if wall[left_x, left_y] == 1:  # corner to south
                right_y = left_y
                while wall[right_x, right_y] == 1:
                    right_x = right_x - 1
                left_x = left_x + 1
                wall_dir = 2
            elif wall[right_x, right_y] == 1:  # corner to north
                left_y = right_y
                while wall[left_x, left_y] == 1:
                    left_x = left_x - 1
                right_x = right_x + 1
                wall_dir = 0
            else:  # wall end
                wall_dir = -1
    max_openings = [(length-2)//12+1 for _, length, _ in wall_blocks]
    if sum(max_openings) < len(opening_area_ids):
        return None
    opening_slot_used = torch.randperm(sum(max_openings))[:len(opening_area_ids)]
    opening_area_ids = torch.tensor(opening_area_ids)
    opening_bounds = ([], [])  # opening bound positions on the left and right side of the wall
    for i, num_slots in enumerate(max_openings):
        ids_to_fit = opening_area_ids[(0 <= opening_slot_used).logical_and(opening_slot_used < num_slots)]
        opening_slot_used = opening_slot_used - num_slots
        block_len = wall_blocks[i][1]
        if ids_to_fit.numel() == 1:
            opening_widths = [torch.randint(low=2, high=min(block_len + 1, 7), size=(1,)).item()]
            start_positions = [torch.randint(low=0, high=block_len-opening_widths[0]+1, size=(1,)).item()]
        elif ids_to_fit.numel() > 1:
            max_opening_space = block_len - (ids_to_fit.numel()-1)*10
            opening_widths = [2]*(ids_to_fit.numel())
            spacings = [0] + [10] * (ids_to_fit.numel() - 1) + [0]
            for _ in range(min(ids_to_fit.numel()*8, max_opening_space-2*ids_to_fit.numel())):
                j = torch.randint(len(opening_widths)+1, size=(1,)).item()
                if j < len(opening_widths) and opening_widths[j] < 10:
                    opening_widths[j] = opening_widths[j] + 1
            for _ in range(block_len - sum(opening_widths) - 10*(ids_to_fit.numel() - 1)):
                j = torch.randint(len(spacings), size=(1,)).item()
                spacings[j] = spacings[j] + 1
            start_positions = [sum(spacings[:j+1]) + sum(opening_widths[:j]) for j in range(len(opening_widths))]
        else:
            opening_widths = []
            start_positions = []
        for start_pos, width, oid in zip(start_positions, opening_widths, ids_to_fit.tolist()):
            # insert openings
            direction, _, (start_left, start_right) = wall_blocks[i]
            if direction == 0:
                wall[start_left[0]+1:start_right[0], start_left[1] - start_pos - width:start_left[1] - start_pos] = oid
                opening_bounds[0].append(((start_left[0] + 1, start_left[1] - start_pos),
                                          (start_left[0] + 1, start_left[1] - start_pos - width)))
                opening_bounds[1].append(((start_right[0] - 1, start_right[1] - start_pos),
                                          (start_right[0] - 1, start_right[1] - start_pos - width)))
            elif direction == 1:
                wall[start_left[0] + start_pos:start_left[0] + start_pos + width, start_left[1]+1:start_right[1]] = oid
                opening_bounds[0].append(((start_left[0] + start_pos, start_left[1] + 1),
                                          (start_left[0] + start_pos + width, start_left[1] + 1)))
                opening_bounds[1].append(((start_right[0] + start_pos, start_right[1] - 1),
                                          (start_right[0] + start_pos + width, start_right[1] - 1)))
            elif direction == 2:
                wall[start_right[0]+1:start_left[0], start_left[1] + start_pos:start_left[1] + start_pos + width] = oid
                opening_bounds[0].append(((start_left[0] - 1, start_left[1] + start_pos),
                                          (start_left[0] - 1, start_left[1] + start_pos + width)))
                opening_bounds[1].append(((start_right[0] + 1, start_right[1] + start_pos),
                                          (start_right[0] + 1, start_right[1] + start_pos + width)))
            elif direction == 3:
                wall[start_left[0] - start_pos - width:start_left[0] - start_pos, start_right[1]+1:start_left[1]] = oid
                opening_bounds[0].append(((start_left[0] - start_pos, start_left[1] - 1),
                                          (start_left[0] - start_pos - width, start_left[1] - 1)))
                opening_bounds[1].append(((start_right[0] - start_pos, start_right[1] + 1),
                                          (start_right[0] - start_pos - width, start_right[1] + 1)))
    return wall, opening_bounds


def add_random_wall(opening_area_ids, room_grid, exit_borders_of_sub_areas):
    max_dist = 9e9
    distance_from_anything = max_dist * torch.ones_like(room_grid)
    distance_from_anything[room_grid != 0] = 0
    has_changed = True
    last_avg_distance = -1
    while has_changed:
        distance_step(distance_from_anything)
        avg_distance = distance_from_anything.mean()
        has_changed = avg_distance != last_avg_distance
        last_avg_distance = avg_distance

    wall_added = False
    tries_left = 5
    while not wall_added:  # might fail so simply starts another try if needed
        # pick random sub-area
        area_id = torch.randint(low=0, high=len(exit_borders_of_sub_areas), size=(1,)).item()
        # find start and end points of wall
        wall_sep_1 = torch.randint(low=0, high=len(exit_borders_of_sub_areas[area_id]), size=(1,)).item()
        wall_sep_2 = torch.randint(low=0, high=len(exit_borders_of_sub_areas[area_id])-1, size=(1,)).item()
        if wall_sep_2 >= wall_sep_1:  # ensure wall_sep_2 != wall_sep_1
            wall_sep_2 += 1
        # wall_seps define which room openings will end up on which side of the new wall
        num_possible_starts = wall_tiles_to_next_opening(
            room_grid,
            exit_borders_of_sub_areas[area_id][(wall_sep_1-1) % len(exit_borders_of_sub_areas[area_id])][1],
            exit_borders_of_sub_areas[area_id][wall_sep_1][0]) - 25
        num_possible_ends = wall_tiles_to_next_opening(
            room_grid,
            exit_borders_of_sub_areas[area_id][(wall_sep_2-1) % len(exit_borders_of_sub_areas[area_id])][1],
            exit_borders_of_sub_areas[area_id][wall_sep_2][0]) - 25
        if num_possible_starts < 1 or num_possible_ends < 1:
            tries_left -= 1
            if tries_left <= 0:
                break  # probably some target areas are too close together to fit a new wall in between
            continue  # there is not enough space between the opening to fit a wall
        wall_start_pos = wall_tiles_to_next_opening(
            room_grid, exit_borders_of_sub_areas[area_id][(wall_sep_1-1) % len(exit_borders_of_sub_areas[area_id])][1],
            num_wall_tiles=torch.randint(low=10, high=10+num_possible_starts, size=(1,)).item())
        wall_end_pos = wall_tiles_to_next_opening(
            room_grid, exit_borders_of_sub_areas[area_id][(wall_sep_2-1) % len(exit_borders_of_sub_areas[area_id])][1],
            num_wall_tiles=torch.randint(low=10, high=10+num_possible_ends, size=(1,)).item())
        thin_wall = draw_thin_wall(wall_start_pos, wall_end_pos, distance_from_anything)
        wall = random_width(thin_wall, wall_start_pos, wall_end_pos, distance_from_anything, room_grid)
        wall_and_openings = insert_openings(wall, wall_start_pos, opening_area_ids)
        if wall_and_openings is not None:
            # draw new wall to room_grid including new openings
            wall, opening_positions = wall_and_openings
            room_grid[wall == 1] = -1  # insert wall
            room_grid[wall > 1] = wall[wall > 1].to(room_grid.dtype)  # insert openings
            # split sub-areas accordingly in exit_borders_of_sub_areas
            old_area_borders = exit_borders_of_sub_areas[area_id]
            if wall_sep_2 > wall_sep_1:  # checked once
                new_border_1 = [(end, start) for start, end in reversed(opening_positions[1])]
                new_border_2 = opening_positions[0]
                old_border_1 = old_area_borders[wall_sep_1:wall_sep_2]
                old_border_2 = old_area_borders[wall_sep_2:] + old_area_borders[:wall_sep_1]
            else:  # checked once
                new_border_1 = opening_positions[0]
                new_border_2 = [(end, start) for start, end in reversed(opening_positions[1])]
                old_border_1 = old_area_borders[wall_sep_2:wall_sep_1]
                old_border_2 = old_area_borders[wall_sep_1:] + old_area_borders[:wall_sep_2]
            new_left_area_borders = new_border_2 + old_border_2
            new_right_area_borders = new_border_1 + old_border_1
            exit_borders_of_sub_areas[area_id] = new_left_area_borders
            exit_borders_of_sub_areas.append(new_right_area_borders)
            wall_added = True
    return wall_added


def setup_random_room():
    """
    Generates a randomized room grid of the size 102x100. All outer borders are either walls or terminal target areas.
    There are up to 10 target areas of which at least 2 are terminal target areas.

    First, the number of terminal target areas n_ta is drawn uniformly from 2,...,4. The respective target areas are
    then placed randomly on the outer walls. They are between 2 and 10 grid tiles wide and the distance between target
    areas is at least 10 tiles.

    Then we consider the graph G with a node for each target area and an edge between nodes i and j if there is a direct
    path between target areas i and j. Thus, we start with a fully connected graph of 2 to 4 nodes.

    To define non-terminal target areas, we start by sampling the number of areas to be placed n_a, uniform random
    integer in [2, 10-n_ta]. Since we will always place target areas in pairs or triples, we proceed by successively
    sampling between pairs and triples until four or less target areas remain. In the resulting succession, we add pairs
    or triples of target areas by picking a maximal fully connected subgraph of G and add a wall with two or three
    openings into the respective area on the grid such that at least one target is separated from the original subgraph.
    The openings are defined as new target areas.

    Walls are 2 to 10 tiles wide and target area openings are 2 to 6 grid tiles long.

    Not implemented: randomly place 0 to 10 walls which may touch another wall on at most one side.

    :return: 2D integer list
    """
    # define target area number permutation
    target_area_numbers = torch.randperm(10)+2
    num_placed_areas = 0
    # define empty room grid
    room_grid = torch.zeros((102, 100), dtype=torch.int)

    # setup terminal target areas
    n_ta = torch.randint(low=2, high=5, size=(1,)).item()
    terminal_target_area_width = torch.randint(low=2, high=11, size=(n_ta,))
    border_tiles = -torch.ones((400,), dtype=torch.int)
    opening_bounds = []

    for taw in terminal_target_area_width:
        area_placed = False
        while not area_placed:
            area_start = torch.randint(high=400-taw, size=(1,)).item()
            blocked_area = (area_start - 10, area_start + taw + 10)
            is_blocked = (
                    torch.cat([border_tiles]*3, dim=0)[400 + blocked_area[0]: 400 + blocked_area[1]] != -1
            ).any()
            if is_blocked:
                area_placed = False
            else:
                # move by one tile if start/end would be exactly in a corner of the grid
                start = (area_start-1) % 400
                end = (area_start + taw.item()) % 400
                corner_pos = {0, 99, 200, 299}
                if start in corner_pos or end in corner_pos:
                    area_start += 1
                    start = (area_start-1) % 400
                    end = (area_start + taw) % 400
                border_tiles[area_start:area_start+taw+1] = target_area_numbers[num_placed_areas]
                opening_bounds.append(
                    ((border_pos_to_2d_pos(start), border_pos_to_2d_pos(end)),
                     area_start)
                )
                num_placed_areas += 1
                area_placed = True
    room_grid[0] = border_tiles[:100]
    room_grid[1:-1, -1] = border_tiles[100:200]
    room_grid[-1] = border_tiles[200:300].flip(0)
    room_grid[1:-1, 0] = border_tiles[300:].flip(0)
    opening_bounds.sort(key=lambda x: x[1], reverse=False)
    opening_bounds = [e[0] for e in opening_bounds]
    # bound positions of a sub-area's exits in counter-clock-wise order
    exit_borders_of_sub_areas = [opening_bounds]

    # define non-terminal target areas
    n_a = torch.randint(low=2, high=11-n_ta, size=(1,)).item()
    wall_openings = []
    while n_a - sum(wall_openings) > 4:
        wall_openings.append(torch.randint(low=2, high=4, size=(1,)).item())
    if n_a - sum(wall_openings) == 4:
        wall_openings = wall_openings + [2, 2]
    elif n_a - sum(wall_openings) == 3:
        wall_openings = wall_openings + [3]
    else:
        wall_openings = wall_openings + [2]
    new_wall = True
    for w_open in wall_openings:
        new_wall = add_random_wall(
            [target_area_numbers[num_placed_areas + i] for i in range(w_open)],
            room_grid,
            exit_borders_of_sub_areas)
        if not new_wall:
            break
        num_placed_areas += w_open
    if not new_wall:
        return setup_random_room()  # attempt failed, start over
    return room_grid
