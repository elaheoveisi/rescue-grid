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
        if d_lava <= 2:
            lava_factor = 0.0   # very near fire
        elif d_lava <= 5:
            lava_factor = 0.5   # near fire
        else:
            lava_factor = 1.0   # far from fire

        d_door = min_dist(door_positions)
        door_factor = 1.0 if d_door <= 2 else 0.0  # near door = slow decay

        dir_mod = self._DIR_MODIFIER.get(victim.direction, 0.0)

        raw = 0.60 * lava_factor + 0.30 * door_factor + 0.10 * dir_mod
        victim.deplete_rate = max(0.5, 2.0 - (5 / 3) * raw)  
        victim.health = max(0.1, min(0.95, 0.95))

    def place_fake_victims(self, level_gen, i, j):
        """Place fake victims in a room using factory pattern."""
        for _ in range(self.num_fake_victims):
            shift = random.choice(self.SHIFTS)
            direction = random.choice(self.DIRECTIONS)
            obj = FakeVictim(shift, direction, color="red")
            level_gen.place_in_room(i, j, obj)

    def _get_room_candidate_positions(self, level_gen, room, lava_positions, near_threshold=5):
        """Return (near_lava, far_from_lava) lists of free positions in a room."""
        top_x, top_y = room.top
        size_x, size_y = room.size
        near, far = [], []
        for y in range(top_y + 1, top_y + size_y - 1):
            for x in range(top_x + 1, top_x + size_x - 1):
                if level_gen.grid.get(x, y) is not None:
                    continue
                d = min(
                    (abs(x - lx) + abs(y - ly) for lx, ly in lava_positions),
                    default=float("inf"),
                )
                if d <= near_threshold:
                    near.append((x, y))
                else:
                    far.append((x, y))
        return near, far

    def place_all(self, level_gen, num_rows, num_cols):
        """Place victims and fake victims in all rooms.

        Half of real victims are placed near lava (distance <= 5),
        half are placed far from lava (distance > 5).
        """
        from minigrid.core.world_object import Door, Lava

        lava_positions = self._collect_positions(level_gen, Lava)
        door_positions = self._collect_positions(level_gen, Door)
        non_important = [d for d in self.DIRECTIONS if d != self.important_victim]

        # How many victims per room should be near vs far lava
        near_target = self.num_real_victims // 2
        far_target = self.num_real_victims - near_target

        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)
                near_positions, far_positions = self._get_room_candidate_positions(
                    level_gen, room, lava_positions
                )
                random.shuffle(near_positions)
                random.shuffle(far_positions)

                # Build placement list: near_target near-lava, far_target far-lava
                # Fall back to the other pool if one is exhausted
                chosen_positions = []
                near_pool = list(near_positions)
                far_pool = list(far_positions)

                for _ in range(near_target):
                    if near_pool:
                        chosen_positions.append(near_pool.pop())
                    elif far_pool:
                        chosen_positions.append(far_pool.pop())

                for _ in range(far_target):
                    if far_pool:
                        chosen_positions.append(far_pool.pop())
                    elif near_pool:
                        chosen_positions.append(near_pool.pop())

                for k in range(self.num_real_victims):
                    direction = (
                        self.important_victim
                        if room.locked
                        else random.choice(non_important)
                    )
                    victim = self._make_victim(direction)

                    if k < len(chosen_positions):
                        pos = chosen_positions[k]
                        level_gen.grid.set(pos[0], pos[1], victim)
                    else:
                        # Fall back to random placement if no positions available
                        _, pos = level_gen.place_in_room(i, j, victim)

                    self._assign_health(
                        victim, pos, lava_positions, door_positions, level_gen
                    )

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
        Place lava tiles in a specific room.

        Args:
            level_gen: The level generator instance
            i: Room row index
            j: Room column index
            num_lava: Number of lava tiles to place (None = use lava_per_room)
        """
        if num_lava is None:
            num_lava = self.lava_per_room

        placed = 0
        failures = 0

        for _ in range(num_lava * 10):
            if placed >= num_lava:
                break
            try:
                level_gen.place_in_room(i, j, Lava())
                placed += 1
                failures = 0
            except Exception:
                failures += 1
                if failures >= 10:
                    break  # Room is likely full

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
