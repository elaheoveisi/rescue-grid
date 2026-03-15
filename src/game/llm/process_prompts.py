from pathlib import Path

import pandas as pd
import yaml

from game.sar.observations import (
    FAKE_VICTIM,
    LAVA,
    VICTIM,
    WALL,
    cam_bounds,
    decode_door,
    decode_key,
    is_door,
    is_key,
)

from .pathfinding import query_all_objects

_DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
_EMPTY = 0
_REWARD = {"victim": "+1.0", "key": "0", "door": "0"}
_SORT_PRIORITY = {
    ("Yes", "Yes"): 0,
    ("Yes", "No"): 1,
    ("No", "Yes"): 2,
    ("No", "No"): 3,
}


def _room(x: int, y: int, room_size: int) -> str:
    rs = max(room_size - 1, 1)
    return f"({y // rs},{x // rs})"


def _build_table(obs: dict) -> str:
    grid = obs["grid"]
    ax, ay = obs["agent_x"], obs["agent_y"]
    room_size = obs["room_size"]
    cx0, cy0, cx1, cy1 = cam_bounds(obs)
    bfs_results = query_all_objects(obs)

    rows = []
    counters: dict[str, int] = {}

    for y, row_cells in enumerate(grid):
        for x, cell in enumerate(row_cells):
            cell = int(cell)
            if cell in (_EMPTY, WALL, LAVA, FAKE_VICTIM) or (x == ax and y == ay):
                continue

            if cell == VICTIM:
                kind, label_base = "victim", "Victim"
            elif cell == FAKE_VICTIM:
                kind, label_base = "fake_victim", "FakeVictim"
            elif is_key(cell):
                kind, label_base = "key", f"{decode_key(cell)['color'].capitalize()}Key"
            elif is_door(cell):
                kind, label_base = (
                    "door",
                    f"{decode_door(cell)['color'].capitalize()}Door",
                )
            else:
                continue

            counters[label_base] = counters.get(label_base, 0) + 1
            pr = bfs_results.get((x, y))
            reachable = "Yes" if (pr and pr.reachable) else "No"
            path_length = (
                pr.path_length if (pr and pr.path_length is not None) else None
            )

            if kind == "door":
                d = decode_door(cell)
                status = (
                    "Open"
                    if d["is_open"]
                    else ("Locked" if d["is_locked"] else "Closed")
                )
                tool = f"{d['color']} key" if d["is_locked"] else "None"
            elif kind == "victim":
                status, tool = "Alive", "None"
            elif kind == "fake_victim":
                status, tool = "Decoy", "None"
            else:
                status, tool = "Available", "None"

            visible = "Yes" if cx0 <= x < cx1 and cy0 <= y < cy1 else "No"
            rows.append(
                {
                    "Object": f"{label_base}{counters[label_base]}",
                    "Room": _room(x, y, room_size),
                    "Visible": visible,
                    "Reachable": reachable,
                    "Reward": _REWARD.get(kind, "0"),
                    "Status": status,
                    "ToolRequired": tool,
                    "PathLength": path_length if path_length is not None else "N/A",
                    "_sort_priority": _SORT_PRIORITY.get((reachable, visible), 3),
                    "_sort_path": path_length if path_length is not None else 9999,
                }
            )

    if not rows:
        return "(no objects on map)"

    df = pd.DataFrame(rows)
    df = df.sort_values(["_sort_priority", "_sort_path"]).drop(
        columns=["_sort_priority", "_sort_path"]
    )
    return df.to_markdown(index=False)


def build_obs(obs: dict) -> str:
    inv = f"{obs['carrying']} key" if obs["carrying"] else "nothing"
    ax, ay = obs["agent_x"], obs["agent_y"]
    room_size = obs["room_size"]
    agent_room = _room(ax, ay, room_size)
    return "\n".join(
        [
            f"Agent: room {agent_room} facing {_DIR_NAMES.get(obs['agent_dir'], '?')} | Carrying: {inv} | Victims remaining: {obs['remaining_victims']} | Steps left: {obs['max_steps'] - obs['step_count']}",
            "",
            "Map objects:",
            _build_table(obs),
        ]
    )


def _build_grid_info(obs: dict) -> str:
    n_rows = obs["num_rows"]
    n_cols = obs["num_cols"]
    mid_r, mid_c = n_rows // 2, n_cols // 2
    return (
        f"There are {n_rows * n_cols} rooms arranged in a {n_rows}×{n_cols} grid. "
        f"({0},{mid_c}) is North, ({n_rows - 1},{mid_c}) is South, "
        f"({mid_r},{0}) is West, ({mid_r},{n_cols - 1}) is East."
    )


def build_prompt(obs: dict, prompt_type: str = "sparse") -> str:
    path = Path(__file__).parent / "prompts.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    suffix_key = "detailed_suffix" if prompt_type == "detailed" else "sparse_suffix"
    template = cfg["preamble"] + cfg[suffix_key]
    return template.format(obs=build_obs(obs), grid_info=_build_grid_info(obs))
