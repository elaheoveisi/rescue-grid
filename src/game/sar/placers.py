import random

from minigrid.core.world_object import Lava

from .objects import REAL_VICTIMS, FakeVictim, Victim


class LockedRoomPlacer:
    """Handles placement of locked rooms and their corresponding keys."""

    def __init__(self, locked_room_prob=0.35):
        self.locked_room_prob = locked_room_prob

    def place_all(self, level_gen, num_rows, num_cols):
        n_locked = max(1, int(num_cols * num_rows * self.locked_room_prob))
        added = 0

        while added < n_locked:
            i, j = level_gen._rand_int(0, num_cols), level_gen._rand_int(0, num_rows)
            locked_room = level_gen.get_room(i, j)

            if locked_room.locked:
                continue

            empty_doors = [idx for idx, d in enumerate(locked_room.doors) if d is None]
            if not empty_doors:
                continue

            door_idx = random.choice(empty_doors)

            if locked_room.neighbors[door_idx] is None:
                continue

            door, _ = level_gen.add_door(i, j, door_idx, locked=True)
            locked_room.locked = True

            while True:
                ki, kj = (
                    level_gen._rand_int(0, num_cols),
                    level_gen._rand_int(0, num_rows),
                )
                key_room = level_gen.get_room(ki, kj)
                if key_room is locked_room:
                    continue
                if getattr(key_room, "locked", False):
                    continue
                level_gen.add_object(ki, kj, "key", door.color)
                break

            added += 1


class VictimPlacer:
    """Handles placement of victims and fake victims."""

    DIRECTIONS = ["up", "down", "left", "right"]
    SHIFTS = ["left", "right"]
    _DIR_MODIFIER = {"up": 0.05, "down": -0.05, "left": 0.0, "right": 0.0}

    def __init__(self, num_fake_victims=3, num_real_victims=1, important_victim="up"):
        """
        Initialize the victim placer.

        Args:
            num_fake_victims: Number of fake victims to place per room
            num_real_victims: Number of real victims to place per room
            important_victim: Direction of the important victim (placed in locked rooms)
        """
        self.num_fake_victims = num_fake_victims
        self.num_real_victims = num_real_victims
        self.important_victim = important_victim

    def _make_victim(self, direction):
        return Victim(direction, color="red")

    def _collect_positions(self, level_gen, obj_type):
        """Scan grid for all instances of obj_type, return list of (x, y)."""
        positions = []
        for y in range(level_gen.height):
            for x in range(level_gen.width):
                obj = level_gen.grid.get(x, y)
                if isinstance(obj, obj_type):
                    positions.append((x, y))
        return positions

    def _assign_health(self, victim, pos, lava_positions, door_positions, level_gen):
        """Set deplete_rate based on direction relative to nearest lava and distance.

        direction == "down": always 4.0 regardless of position.

        Otherwise, classify victim direction as toward / perpendicular / away
        from the nearest lava tile, then look up rate by distance tier:

          distance        toward  perp  away
          near  (d<=2)     3.0   2.0   1.5
          medium(2<d<=5)   1.5   1.3   1.1
          far/near-door    0.5   0.5   0.5
        """
        if getattr(victim, "direction", None) == "down":
            victim.deplete_rate = 6.0
            victim.health = 0.60
            return

        max_dist = (level_gen.width + level_gen.height) / 2

        def min_dist(targets):
            if not targets:
                return max_dist
            return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

        d_door = min_dist(door_positions)

        if not lava_positions:
            victim.deplete_rate = 0.5
            victim.health = 0.90
            return

        rates = {
            "toward": {"near": 4, "medium": 2.75, "safe": 1, "door": 0.5},
            "perp":   {"near": 2.75, "medium": 1.3, "safe": 0.5, "door": 0.5},
            "away":   {"near": 1.5, "medium": 1.1, "safe": 0.25, "door": 0.05},
        }

        nearest_lava = min(lava_positions, key=lambda t: abs(pos[0] - t[0]) + abs(pos[1] - t[1]))
        d = abs(pos[0] - nearest_lava[0]) + abs(pos[1] - nearest_lava[1])

        if d_door <= 2:
            tier = "door"
        elif d > 5:
            tier = "safe"
        elif d <= 2:
            tier = "near"
        else:
            tier = "medium"

        dx = nearest_lava[0] - pos[0]
        dy = nearest_lava[1] - pos[1]
        direction = getattr(victim, "direction", None)
        if abs(dx) >= abs(dy):
            toward_dir, away_dir = ("right", "left") if dx > 0 else ("left", "right")
        else:
            toward_dir, away_dir = ("down", "up") if dy > 0 else ("up", "down")

        if direction == toward_dir:
            orientation = "toward"
        elif direction == away_dir:
            orientation = "away"
        else:
            orientation = "perp"

        victim.deplete_rate = rates[orientation][tier]
        starting_health = {"down": 0.60, "up": 0.90, "left": 0.75, "right": 0.75}
        victim.health = starting_health.get(getattr(victim, "direction", None), 0.90)

    def place_fake_victims(self, level_gen, i, j):
        """Place fake victims in a room using factory pattern."""
        for _ in range(self.num_fake_victims):
            shift = random.choice(self.SHIFTS)
            direction = random.choice(self.DIRECTIONS)
            obj = FakeVictim(shift, direction, color="red")
            level_gen.place_in_room(i, j, obj)

    def place_all(self, level_gen, num_rows, num_cols):
        """Place victims spread evenly across each room with 25% per direction.

        Victims are placed using a sector grid so they cover the full room.
        Directions are shuffled per room: 25% up, 25% down, 25% left, 25% right.
        Locked rooms override all directions to important_victim.
        """
        from minigrid.core.world_object import Door, Lava

        lava_positions = self._collect_positions(level_gen, Lava)
        door_positions = self._collect_positions(level_gen, Door)

        n = self.num_real_victims
        per_dir = n // 4
        leftover = n - 4 * per_dir

        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)
                top_x, top_y = room.top
                size_x, size_y = room.size
                inner_w = size_x - 2
                inner_h = size_y - 2

                # 25% each direction, shuffled; locked rooms all use important_victim
                if room.locked:
                    directions = [self.important_victim] * n
                else:
                    dirs = self.DIRECTIONS * per_dir + random.choices(self.DIRECTIONS, k=leftover)
                    random.shuffle(dirs)
                    directions = dirs

                # Pick up to 2 free cells near a door (manhattan distance <= 2)
                n_door = 2
                door_candidates = [
                    (x, y)
                    for y in range(top_y + 1, top_y + size_y - 1)
                    for x in range(top_x + 1, top_x + size_x - 1)
                    if level_gen.grid.get(x, y) is None
                    and min(
                        (abs(x - dx) + abs(y - dy) for dx, dy in door_positions),
                        default=float("inf"),
                    ) <= 2
                ]
                random.shuffle(door_candidates)
                door_spots = door_candidates[:n_door]
                reserved = set(door_spots)

                # Place door victims (first n_door directions from the shuffled list)
                for k, pos in enumerate(door_spots):
                    victim = self._make_victim(directions[k])
                    level_gen.grid.set(pos[0], pos[1], victim)
                    self._assign_health(victim, pos, lava_positions, door_positions, level_gen)

                # Place remaining victims in sectors, skipping reserved cells
                n_sector = n - len(door_spots)
                cols_s = max(1, round(((inner_w / inner_h) * n_sector) ** 0.5))
                rows_s = max(1, (n_sector + cols_s - 1) // cols_s)
                sx = inner_w / cols_s
                sy = inner_h / rows_s

                sectors = [(c, r) for c in range(cols_s) for r in range(rows_s)]
                random.shuffle(sectors)
                sectors = sectors[:n_sector]

                for (c, r), direction in zip(sectors, directions[len(door_spots):]):
                    x0 = top_x + 1 + int(c * sx)
                    x1 = top_x + 1 + (inner_w if c == cols_s - 1 else int((c + 1) * sx))
                    y0 = top_y + 1 + int(r * sy)
                    y1 = top_y + 1 + (inner_h if r == rows_s - 1 else int((r + 1) * sy))

                    candidates = [
                        (x, y)
                        for x in range(x0, x1)
                        for y in range(y0, y1)
                        if level_gen.grid.get(x, y) is None and (x, y) not in reserved
                    ]

                    victim = self._make_victim(direction)
                    if candidates:
                        pos = random.choice(candidates)
                        level_gen.grid.set(pos[0], pos[1], victim)
                    else:
                        _, pos = level_gen.place_in_room(i, j, victim)

                    self._assign_health(victim, pos, lava_positions, door_positions, level_gen)

                self.place_fake_victims(level_gen, i, j)


class VictimTracker:
    """Tracks alive victims, handles health decay and battery display."""

    def __init__(self):
        self._positions = []  # list of (x, y, obj)

    def initialize(self, grid, width, height):
        self._positions = [
            (x, y, grid.get(x, y))
            for y in range(height)
            for x in range(width)
            if isinstance(grid.get(x, y), REAL_VICTIMS)
        ]

    @property
    def count(self):
        return len(self._positions)

    def sync_after_pickup(self, grid):
        self._positions = [
            (x, y, obj) for x, y, obj in self._positions if grid.get(x, y) is obj
        ]

    def show_visible_batteries(
        self, camera, grid_width, grid_height, seconds: float = 10.0
    ):
        x0, y0, x1, y1 = camera.get_visible_bounds(grid_width, grid_height)
        for x, y, obj in self._positions:
            if x0 <= x < x1 and y0 <= y < y1:
                obj.show_battery(seconds)

    def hide_all_batteries(self):
        for _, _, obj in self._positions:
            obj.hide_battery()

    def decay(self, camera, grid, grid_width, grid_height, deplete_amount):
        x0, y0, x1, y1 = camera.get_visible_bounds(grid_width, grid_height)
        for x, y, obj in self._positions:
            if x0 <= x < x1 and y0 <= y < y1:
                obj.deplete(deplete_amount)


class LavaPlacer:
    """Handles placement of lava obstacles in the environment."""

    def __init__(self, lava_per_room=0, lava_probability=0.3, enabled=True):
        """
        Initialize lava placer.

        Args:
            lava_per_room: Fixed number of lava tiles per room (0 = use probability)
            lava_probability: Probability of placing lava in each room (used if lava_per_room=0)
            enabled: If False, place_all is a no-op
        """
        self.lava_per_room = lava_per_room
        self.lava_probability = lava_probability
        self.enabled = enabled

    def place_in_room(self, level_gen, i, j, num_lava=None):
        """
        Place lava tiles evenly distributed across a room by dividing it into sectors.
        One lava tile is placed in a random free cell within each sector.
        Lava tiles will not be adjacent to each other or to doors.

        Args:
            level_gen: The level generator instance
            i: Room row index
            j: Room column index
            num_lava: Number of lava tiles to place (None = use lava_per_room)
        """
        from minigrid.core.world_object import Door

        if num_lava is None:
            num_lava = self.lava_per_room

        room = level_gen.get_room(i, j)
        top_x, top_y = room.top
        size_x, size_y = room.size

        inner_w = size_x - 2  # exclude border walls
        inner_h = size_y - 2

        # Cells forbidden for lava: adjacent to any door in this room
        forbidden = set()
        for bx in range(top_x, top_x + size_x):
            for by in range(top_y, top_y + size_y):
                if isinstance(level_gen.grid.get(bx, by), Door):
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        forbidden.add((bx + dx, by + dy))

        # Divide interior into a grid of num_lava sectors (roughly square)
        cols = max(1, round(((inner_w / inner_h) * num_lava) ** 0.5))
        rows = max(1, (num_lava + cols - 1) // cols)

        sx = inner_w / cols
        sy = inner_h / rows

        sectors = [(c, r) for c in range(cols) for r in range(rows)]
        random.shuffle(sectors)

        placed = 0
        for c, r in sectors:
            if placed >= num_lava:
                break
            x0 = top_x + 1 + int(c * sx)
            x1 = top_x + 1 + (inner_w if c == cols - 1 else int((c + 1) * sx))
            y0 = top_y + 1 + int(r * sy)
            y1 = top_y + 1 + (inner_h if r == rows - 1 else int((r + 1) * sy))

            candidates = [
                (x, y)
                for x in range(x0, x1)
                for y in range(y0, y1)
                if level_gen.grid.get(x, y) is None and (x, y) not in forbidden
            ]
            if candidates:
                x, y = random.choice(candidates)
                level_gen.grid.set(x, y, Lava())
                placed += 1
                # Forbid all 8 neighbors so lava tiles are never adjacent
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    forbidden.add((x + dx, y + dy))

    def place_all(self, level_gen, num_rows, num_cols, skip_locked_rooms=False):
        """
        Place lava in all rooms based on configuration.

        Args:
            level_gen: The level generator instance
            num_rows: Number of room rows
            num_cols: Number of room columns
            skip_locked_rooms: If True, don't place lava in locked rooms
        """
        if not self.enabled:
            return

        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)

                # Skip locked rooms if requested
                if skip_locked_rooms and getattr(room, "locked", False):
                    continue

                # Decide whether to place lava in this room
                if self.lava_per_room > 0:
                    # Fixed number per room
                    self.place_in_room(level_gen, i, j, self.lava_per_room)
                elif random.random() < self.lava_probability:
                    # Random placement based on probability
                    num_lava = random.randint(1, 3)  # 1-3 lava tiles
                    self.place_in_room(level_gen, i, j, num_lava)
