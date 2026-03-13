from pathlib import Path

import yaml
from game.sar.observations import FAKE_VICTIM, LAVA, VICTIM, cam_bounds, decode_door, is_door, is_key

_DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
_DIR_CHARS = {0: ">", 1: "v", 2: "<", 3: "^"}
# (dx, dy) for the cell directly ahead per facing direction
_DIR_AHEAD = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_WALL = 1


def _door_symbol(cell: int) -> str:
    d = decode_door(cell)
    if d["is_open"]:
        return "O"
    if d["is_locked"]:
        return "L"
    return "D"


def _front_cell_desc(obs: dict) -> str:
    dx, dy = _DIR_AHEAD[obs["agent_dir"]]
    fx, fy = obs["agent_x"] + dx, obs["agent_y"] + dy
    grid = obs["grid"]
    if not (0 <= fy < len(grid) and 0 <= fx < len(grid[0])):
        return "wall"
    cell = int(grid[fy][fx])
    if cell == 0:
        return "empty"
    if cell == _WALL:
        return "wall"
    if cell == LAVA:
        return "lava"
    if cell == VICTIM:
        return "victim"
    if cell == FAKE_VICTIM:
        return "fake victim"
    if is_key(cell):
        return "key"
    if is_door(cell):
        d = decode_door(cell)
        state = "open" if d["is_open"] else ("locked" if d["is_locked"] else "closed")
        return f"{d['color']} door [{state}]"
    return "unknown"


def _camera_view(obs: dict) -> str:
    grid = obs["grid"]
    ax, ay = obs["agent_x"], obs["agent_y"]
    adir = obs["agent_dir"]
    cx0, cy0, cx1, cy1 = cam_bounds(obs)

    rows = []
    for y in range(cy0, cy1):
        row = []
        for x in range(cx0, cx1):
            if x == ax and y == ay:
                row.append(_DIR_CHARS.get(adir, "A"))
                continue
            if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
                row.append("#")
                continue
            cell = int(grid[y][x])
            if cell == 0:
                row.append("E")
            elif cell == _WALL:
                row.append("W")
            elif cell == LAVA:
                row.append("R")
            elif cell == VICTIM:
                row.append("V")
            elif cell == FAKE_VICTIM:
                row.append("F")
            elif is_key(cell):
                row.append("K")
            elif is_door(cell):
                row.append(_door_symbol(cell))
            else:
                row.append("E")
        rows.append("".join(row))
    return "\n".join(rows)


def build_obs(obs: dict) -> str:
    carrying = obs["carrying"]
    inv = f"{carrying} key" if carrying else "nothing"
    ax, ay = obs["agent_x"], obs["agent_y"]
    lines = [
        f"Agent: ({ax},{ay}) facing {_DIR_NAMES.get(obs['agent_dir'], '?')}  |  Carrying: {inv}  |  Victims remaining: {obs['remaining_victims']}",
        f"Front cell: {_front_cell_desc(obs)}",
        "",
        "Camera view (north=up):",
        "Legend: W wall  E empty  > v < ^ agent  V victim  F fake  K key  L locked-door  D closed-door  O open-door  R lava",
        _camera_view(obs),
    ]
    return "\n".join(lines)


def build_prompt(obs: dict, prompt_type: str = "decompose") -> str:
    path = Path(__file__).parent / "prompts.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["prompt"].format(obs=build_obs(obs))
