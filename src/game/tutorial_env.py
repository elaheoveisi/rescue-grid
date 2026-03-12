import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Door, Key
from minigrid.envs.babyai.core.verifier import ObjDesc, OpenInstr, PickupInstr

from .core.level import SARLevelGen
from .sar.actions import RescueAction


class TutorialEnv(SARLevelGen):
    def __init__(self, config=None, **kwargs):
        config = config or {}
        kwargs.setdefault("num_rows", 1)
        kwargs.setdefault("num_cols", 1)
        super().__init__(**kwargs)
        self.resuce_action = RescueAction(self)

    def gen_mission(self):
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        target = self._place_random_objects()

        if isinstance(target, Door):
            self.mission = f"Open the {target.color} door"
            self.instrs = OpenInstr(ObjDesc(target.type, target.color))
        else:
            self.mission = f"Pick up the {target.color} {target.type}"
            self.instrs = PickupInstr(ObjDesc(target.type, target.color))

        print(self.mission)

    def _place_random_objects(self):
        # Always place a door — random color, randomly locked or unlocked
        door_color = random.choice(COLOR_NAMES)
        locked = random.choice([True, False])
        door = Door(door_color, is_locked=locked)

        # Pick a random wall (0=right, 1=bottom, 2=left, 3=top) and position along it
        wall = random.randint(0, 3)
        if wall == 0:
            x, y = self.width - 1, random.randint(1, self.height - 2)
        elif wall == 1:
            x, y = random.randint(1, self.width - 2), self.height - 1
        elif wall == 2:
            x, y = 0, random.randint(1, self.height - 2)
        else:
            x, y = random.randint(1, self.width - 2), 0
        self.grid.set(x, y, door)

        # If locked, place a matching key so the agent can open it
        if locked:
            self.place_in_room(0, 0, Key(door_color))

        # Pick a key color different from the door color to avoid ambiguity
        other_colors = [c for c in COLOR_NAMES if c != door_color]
        key_color = random.choice(other_colors)

        # Randomly select a mission target: a key, a ball, or the door itself
        for i in range(5):
            target = random.choice([Key(key_color), Ball(), door])
            if target is not door:
                self.place_in_room(0, 0, target)

        return target

    def get_mission_status(self):
        status = "tutorial"
        return {
            "saved_victims": 0,
            "remaining_victims": 0,
            "status": status,
        }
