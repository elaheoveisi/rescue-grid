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
        """Set victim health based on proximity to lava and doors.

        Near fire thresholds:
          - very near fire: distance <= 2  → lava_factor = 0.0 (fastest decay)
          - near fire:      distance 3–5   → lava_factor = 0.5
          - far from fire:  distance > 5   → lava_factor = 1.0 (slowest decay)

        Near door threshold:
          - near door: distance <= 2 → door_factor = 1.0 (slowest decay)
          - far door:  distance > 2  → door_factor = 0.0
        """
        max_dist = (level_gen.width + level_gen.height) / 2

        def min_dist(targets):
            if not targets:
                return max_dist
            return min(abs(pos[0] - tx) + abs(pos[1] - ty) for tx, ty in targets)

        d_lava = min_dist(lava_positions)
        d_door = min_dist(door_positions)
        if d_lava <= 2:
            victim.deplete_rate = 3.0
        elif d_door <= 2 or d_lava > 5:
            victim.deplete_rate = 0.5
        else:
            victim.deplete_rate = 1.3
        victim.health = max(0.1, min(0.95, 0.95))

    def place_fake_victims(self, level_gen, i, j):
        """Place fake victims in a room using factory pattern."""
        for _ in range(self.num_fake_victims):
            shift = random.choice(self.SHIFTS)
            direction = random.choice(self.DIRECTIONS)
            obj = FakeVictim(shift, direction, color="red")
            level_gen.place_in_room(i, j, obj)

    def _get_room_candidate_positions(self, level_gen, room, lava_positions, door_positions):
        """Classify free cells in a room into three tiers matching the health decay rates.

        near:   d_lava <= 2               → fast decay (rate 2.0)
        middle: 2 < d_lava <= 5           → medium decay (rate 1.5)
        safe:   d_lava > 5 or d_door <= 2 → slow decay (rate 0.5)
        """
        top_x, top_y = room.top
        size_x, size_y = room.size
        near, middle, safe = [], [], []
        for y in range(top_y + 1, top_y + size_y - 1):
            for x in range(top_x + 1, top_x + size_x - 1):
                if level_gen.grid.get(x, y) is not None:
                    continue
                d_lava = min(
                    (abs(x - lx) + abs(y - ly) for lx, ly in lava_positions),
                    default=float("inf"),
                )
                d_door = min(
                    (abs(x - dx) + abs(y - dy) for dx, dy in door_positions),
                    default=float("inf"),
                )
                if d_lava <= 2:
                    near.append((x, y))
                elif d_door <= 2 or d_lava > 5:
                    safe.append((x, y))
                else:
                    middle.append((x, y))
        return near, middle, safe

    def place_all(self, level_gen, num_rows, num_cols):
        """Place victims with ~33% in each distance tier (near/middle/safe lava)."""
        from minigrid.core.world_object import Door, Lava

        lava_positions = self._collect_positions(level_gen, Lava)
        door_positions = self._collect_positions(level_gen, Door)
        non_important = [d for d in self.DIRECTIONS if d != self.important_victim]

        # Build a shuffled tier sequence so ~1/3 of all victims land in each zone
        total = num_rows * num_cols * self.num_real_victims
        n = total // 3
        tier_sequence = ["near"] * n + ["middle"] * n + ["safe"] * (total - 2 * n)
        random.shuffle(tier_sequence)
        tier_idx = 0

        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)
                near, middle, safe = self._get_room_candidate_positions(
                    level_gen, room, lava_positions, door_positions
                )
                pools = {"near": near, "middle": middle, "safe": safe}

                for _ in range(self.num_real_victims):
                    preferred = tier_sequence[tier_idx]
                    tier_idx += 1
                    pool = pools[preferred] or near or middle or safe

                    direction = self.important_victim if room.locked else random.choice(non_important)
                    victim = self._make_victim(direction)

                    if pool:
                        pos = random.choice(pool)
                        pool.remove(pos)
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
            x1 = top_x + 1 + min(int((c + 1) * sx), inner_w)
            y0 = top_y + 1 + int(r * sy)
            y1 = top_y + 1 + min(int((r + 1) * sy), inner_h)

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
