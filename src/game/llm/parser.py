from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from game.sar.observations import VICTIM, cam_bounds, decode_door, decode_key, is_door, is_key


# ---------------------------------------------------------------------------
# Goal types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RescueVictim:
    def __str__(self) -> str:
        return "There are some victims near by rescue them"


@dataclass(frozen=True)
class GetKey:
    color: str

    def __str__(self) -> str:
        return f"Get the {self.color} key may be we can use it to open {self.color} door"


@dataclass(frozen=True)
class ClearDoor:
    color: str

    def __str__(self) -> str:
        return f"Open the {self.color} door"


@dataclass(frozen=True)
class DropKey:
    def __str__(self) -> str:
        return "Drop the key"


Goal = Union[RescueVictim, GetKey, ClearDoor, DropKey]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_GET_KEY_RE = re.compile(r"GetKey\s*\(\s*(\w+)\s*\)", re.IGNORECASE)
_CLEAR_DOOR_RE = re.compile(r"ClearDoor\s*\(\s*(\w+)\s*\)", re.IGNORECASE)
_RESCUE_RE = re.compile(r"RescueVictim", re.IGNORECASE)
_DROP_RE = re.compile(r"DropKey", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Visibility helper
# ---------------------------------------------------------------------------

def _visible_objects(obs: dict) -> tuple[set[str], set[str], bool]:
    """Return (key_colors, door_colors, victim_visible) from the camera view."""
    x0, y0, x1, y1 = cam_bounds(obs)

    grid = obs["grid"]
    key_colors: set[str] = set()
    door_colors: set[str] = set()
    victim_visible = False

    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            cell = int(cell)
            if cell == VICTIM:
                victim_visible = True
            elif is_key(cell):
                key_colors.add(decode_key(cell)["color"])
            elif is_door(cell):
                door_colors.add(decode_door(cell)["color"])

    return key_colors, door_colors, victim_visible


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_goals(response: str, obs: dict | None = None) -> list[Goal]:
    """Extract and validate the goal sequence from an LLM response.

    Looks for a <START>...<END> block. When obs is provided, goals that
    reference objects not currently visible are dropped with a warning.
    """
    match = re.search(r"<START>(.*?)<END>", response, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    vis_keys, vis_doors, victim_vis = _visible_objects(obs) if obs else (None, None, None)

    goals: list[Goal] = []
    for raw in match.group(1).splitlines():
        if goals:  # stop after the first valid goal
            break
        line = raw.strip()
        if not line:
            continue

        m = _GET_KEY_RE.search(line)
        if m:
            color = m.group(1).lower()
            if vis_keys is not None and color not in vis_keys:
                print(f"[parser] Dropping GetKey({color}): not visible")
                continue
            goals.append(GetKey(color))
            continue

        m = _CLEAR_DOOR_RE.search(line)
        if m:
            color = m.group(1).lower()
            if vis_doors is not None and color not in vis_doors:
                print(f"[parser] Dropping ClearDoor({color}): not visible")
                continue
            goals.append(ClearDoor(color))
            continue

        if _RESCUE_RE.search(line):
            if victim_vis is not None and not victim_vis:
                print("[parser] Dropping RescueVictim: no victim visible")
                continue
            goals.append(RescueVictim())
            continue

        if _DROP_RE.search(line):
            goals.append(DropKey())
            continue

    return goals
