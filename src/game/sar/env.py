import random

from minigrid.core.world_object import Door

from ..core.level import SARLevelGen
from .actions import RescueAction
from .instructions import PickupAllVictimsInstr, calculate_max_steps
from .objects import REAL_VICTIMS
from .observations import GameObservation
from .utils import LavaPlacer, VictimPlacer


def build_sar_env(
    screen_size: int,
    num_rows: int = 3,
    num_cols: int = 3,
    num_fake_victims: int = 5,
    num_real_victims: int = 2,
    important_victim: str = "down",
    lava_per_room: int = 2,
    locked_room_prob: float = 0.35,
    tile_size: int = 64,
    **kwargs,
) -> "PickupVictimEnv":
    """Factory that creates a fully configured PickupVictimEnv."""
    victim_placer = VictimPlacer(
        num_fake_victims=num_fake_victims,
        num_real_victims=num_real_victims,
        important_victim=important_victim,
    )
    return PickupVictimEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        screen_size=screen_size,
        render_mode="rgb_array",
        agent_pov=True,
        add_lava=True,
        lava_per_room=lava_per_room,
        locked_room_prob=locked_room_prob,
        tile_size=tile_size,
        victim_placer=victim_placer,
        **kwargs,
    )


class PickupVictimEnv(SARLevelGen):
    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        unblocking=False,
        add_lava=True,
        lava_per_room=0,
        lava_probability=0.5,
        locked_room_prob=0.5,
        victim_placer=None,
        **kwargs,
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            locations=False,
            unblocking=unblocking,  # Configurable: False ensures all victims are reachable
            implicit_unlock=False,
            **kwargs,
        )
        self.victim_placer = victim_placer
        self.locked_room_prob = locked_room_prob

        # Lava configuration
        self.add_lava = add_lava
        if add_lava:
            self.lava_placer = LavaPlacer(
                lava_per_room=lava_per_room, lava_probability=lava_probability
            )

        # Custom actions
        self.rescue_action = RescueAction(self, fallback=self._step)
        self.observation = GameObservation()

    def add_locked_rooms(self, n_locked):
        added = 0

        while added < n_locked:
            # Pick a random room
            i, j = self._rand_int(0, self.num_cols), self._rand_int(0, self.num_rows)
            locked_room = self.get_room(i, j)

            # Skip if room is already locked
            if locked_room.locked:
                continue

            # Find all door indices that are still empty
            empty_doors = [idx for idx, d in enumerate(locked_room.doors) if d is None]
            if not empty_doors:
                continue  # no free door, pick another room

            # Pick a random empty door
            door_idx = random.choice(empty_doors)

            # Skip if door leads outside
            if locked_room.neighbors[door_idx] is None:
                continue

            # Add locked door
            door, _ = self.add_door(i, j, door_idx, locked=True)
            locked_room.locked = True

            # Place key in a different room
            while True:
                ki, kj = (
                    self._rand_int(0, self.num_cols),
                    self._rand_int(0, self.num_rows),
                )
                key_room = self.get_room(ki, kj)
                if key_room is locked_room:
                    continue
                if getattr(key_room, "locked", False):
                    continue  # avoid placing key in another locked room
                self.add_object(ki, kj, "key", door.color)
                break

            added += 1

    def _count_objects_by_type(self, obj_types):
        return len(self._find_objects_by_type(obj_types))

    def _find_objects_by_type(self, obj_types):
        objects = []
        for y in range(self.height):
            for x in range(self.width):
                obj = self.grid.get(x, y)
                if isinstance(obj, obj_types):
                    objects.append(obj)
        return objects

    def get_all_victims(self):
        """
        Returns a list of all victim objects currently present in the environment.
        """
        return self._find_objects_by_type(REAL_VICTIMS)

    def get_mission_status(self):
        """
        Returns the current mission status.

        Returns:
            dict: Dictionary containing:
                - 'status': 'success', 'failure', or 'incomplete'
                - 'saved_victims': Number of victims saved
                - 'total_victims': Total number of victims in the mission
                - 'remaining_victims': Number of victims left to save
        """
        if hasattr(self, "instrs") and self.instrs is not None:
            status = self.instrs.verify(self)
        else:
            status = "incomplete"

        return {
            "status": status,
            "saved_victims": self.saved_victims,
            "remaining_victims": getattr(self, "total_victims", 0) - self.saved_victims,
        }

    def validate_instrs(self, instrs):
        """
        Override to handle custom instruction types.

        Args:
            instrs: Instruction object to validate
        """
        # Allow our custom PickupAllVictimsInstr without validation
        if isinstance(instrs, PickupAllVictimsInstr):
            return

        # Fall back to parent validation for standard instructions
        super().validate_instrs(instrs)

    def num_navs_needed(self, instrs):
        """
        Override to handle custom instruction types.

        Args:
            instrs: Instruction object

        Returns:
            int: Number of navigation actions needed (for mission generation)
        """
        # For PickupAllVictimsInstr, return number of victims
        if isinstance(instrs, PickupAllVictimsInstr):
            return instrs.num_victims

        # Fall back to parent method for standard instructions
        return super().num_navs_needed(instrs)

    def reset(self, **kwargs):
        """Reset the environment and all stats."""
        self.saved_victims = 0
        self.fixed_max_steps = calculate_max_steps(
            room_size=self.room_size,
            num_cols=self.num_cols,
            num_rows=self.num_rows,
            victims_per_room=self.victim_placer.num_real_victims,
            num_doors=self._count_objects_by_type(Door),
        )
        self.max_steps = self.fixed_max_steps
        self._deplete_amount = 1.0 / self.fixed_max_steps
        obs, info = super().reset(**kwargs)
        self.instrs.reset_verifier(self)
        self.total_victims = self._count_objects_by_type(REAL_VICTIMS)
        self._victim_positions = [
            (x, y, self.grid.get(x, y))
            for y in range(self.height)
            for x in range(self.width)
            if isinstance(self.grid.get(x, y), REAL_VICTIMS)
        ]
        self.camera.reset()
        obs = self.observation.process_observation(obs, self)
        return obs, info

    def gen_mission(self):
        """Generate the mission layout and instructions."""

        # Add locked rooms (20% of rooms - balanced between challenge and generation speed)
        n_locked = max(1, int(self.num_cols * self.num_rows * self.locked_room_prob))
        self.add_locked_rooms(n_locked)

        self.connect_all()

        # Add lava obstacles (before victims to avoid blocking them)
        if self.add_lava:
            self.lava_placer.place_all(self, self.num_rows, self.num_cols)

        # Place agent outside locked room
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            if not start_room.locked:
                break

        # Check that all objects (including victims) are reachable from agent start position
        if not self.unblocking:
            self.check_objs_reachable()

        # Add victims after checking reachability
        self.victim_placer.place_all(self, self.num_rows, self.num_cols)

        victims = self.get_all_victims()

        # Create instruction to pick up all victims
        self.instrs = PickupAllVictimsInstr(victims)

    def _step(self, action):
        return super().step(action)

    def step(self, action):
        if action == self.actions.pickup:
            obs, reward, terminated, truncated, info = self.rescue_action.execute()
            self._victim_positions = [
                (x, y, obj) for x, y, obj in self._victim_positions
                if self.grid.get(x, y) is obj
            ]

            # Verify if mission is complete after pickup action
            if hasattr(self, "instrs") and self.instrs is not None:
                status = self.instrs.verify(self)
                if status == "success":
                    terminated = True
                    reward += 1.0  # Bonus reward for completing mission
                    info["mission_complete"] = True

        else:
            obs, reward, terminated, truncated, info = self._step(action)

        x0, y0, x1, y1 = self.camera.get_visible_bounds(self.width, self.height)
        for x, y, obj in self._victim_positions:
            if x0 <= x < x1 and y0 <= y < y1:
                obj.deplete(self._deplete_amount)

        obs = self.observation.process_observation(obs, self)
        return obs, reward, terminated, truncated, info
