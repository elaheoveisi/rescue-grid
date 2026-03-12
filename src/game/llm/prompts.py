from pathlib import Path

import yaml
from game.sar.observations import (
    FAKE_VICTIM,
    LAVA,
    VICTIM,
    decode_door,
    decode_key,
    is_door,
    is_key,
)

_DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
_DIR_CHARS = {0: ">", 1: "v", 2: "<", 3: "^"}
# (ahead_dx, ahead_dy, left_dx, left_dy) per facing direction
_DIR_VECTORS = {
    0: (1, 0, 0, -1),  # East
    1: (0, 1, 1, 0),  # South
    2: (-1, 0, 0, 1),  # West
    3: (0, -1, -1, 0),  # North
}
# (ahead, behind, left, right) compass names per facing direction
_DIR_SIDE_NAMES = {
    0: ("East", "West", "North", "South"),
    1: ("South", "North", "East", "West"),
    2: ("West", "East", "South", "North"),
    3: ("North", "South", "West", "East"),
}
_CELL_SYMBOLS = {0: ".", LAVA: "~", VICTIM: "V", FAKE_VICTIM: "F"}


def _cell_symbol(cell: int) -> str:
    if cell in _CELL_SYMBOLS:
        return _CELL_SYMBOLS[cell]
    if cell == 1:
        return "#"
    if is_door(cell):
        return "D"
    if is_key(cell):
        return "K"
    return "?"


def _load_prompts() -> dict:
    path = Path(__file__).parent / "prompts.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _compose(shared: dict, reasoning: str, output: str) -> str:
    parts = [
        shared["preamble"].rstrip(),
        reasoning.rstrip(),
        shared["objectives"].rstrip(),
        output.rstrip(),
        shared["footer"].rstrip(),
    ]
    return "\n\n".join(parts)


_cfg = _load_prompts()
_shared = _cfg["shared"]
_PROMPT_TEMPLATES: dict[str, str] = {
    name: _compose(_shared, spec["reasoning"], spec["output"])
    for name, spec in _cfg["prompts"].items()
}


def _cam_bounds(obs: dict):
    """Return (x0, y0, x1, y1) of the camera view, falling back to half=6."""
    if "cam_top_x" in obs:
        x0 = obs["cam_top_x"]
        y0 = obs["cam_top_y"]
        return x0, y0, x0 + obs["cam_view_w"], y0 + obs["cam_view_h"]
    half = 6
    ax, ay = obs["agent_x"], obs["agent_y"]
    return ax - half, ay - half, ax + half + 1, ay + half + 1


def _rel_pos(ax: int, ay: int, ox: int, oy: int) -> str:
    """Compass offset of (ox, oy) from agent at (ax, ay)."""
    dx, dy = ox - ax, oy - ay
    parts = []
    if dy < 0:
        parts.append(f"{-dy}N")
    elif dy > 0:
        parts.append(f"{dy}S")
    if dx > 0:
        parts.append(f"{dx}E")
    elif dx < 0:
        parts.append(f"{-dx}W")
    return "".join(parts) if parts else "here"


def _render(obs: dict) -> tuple[str, str, str]:
    """Convert int grid → symbol grid, slice for local view, scan for legend."""
    ax, ay = obs["agent_x"], obs["agent_y"]
    grid = obs["grid"]
    x0, y0, x1, y1 = _cam_bounds(obs)

    # Symbol grid: comprehension over int grid, then stamp agent position
    syms = [[_cell_symbol(int(c)) for c in row] for row in grid]
    syms[ay][ax] = _DIR_CHARS.get(obs["agent_dir"], "A")

    # Both maps are slices/joins of the same array — no extra grid scan
    full_map = "\n".join("".join(row) for row in syms)
    local_view = "\n".join("".join(row[x0:x1]) for row in syms[y0:y1])

    # Legend: scan int grid, skip trivial cells (empty=0, wall=1)
    victims, fakes, lavas, doors, keys = [], [], [], [], []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            cell = int(cell)
            if cell <= 1:
                continue
            rel = _rel_pos(ax, ay, x, y)
            if cell == LAVA:
                lavas.append(rel)
            elif cell == VICTIM:
                victims.append(rel)
            elif cell == FAKE_VICTIM:
                fakes.append(rel)
            elif is_door(cell):
                d = decode_door(cell)
                state = (
                    "open"
                    if d["is_open"]
                    else ("locked" if d["is_locked"] else "closed")
                )
                doors.append(f"  {d['color']} door [{state}]: {rel}")
            elif is_key(cell):
                keys.append(f"  {decode_key(cell)['color']} key: {rel}")

    legend_lines = []
    if doors:
        legend_lines += ["Doors:"] + doors
    if keys:
        legend_lines += ["Keys:"] + keys
    if victims:
        legend_lines.append("Victims: " + ", ".join(victims))
    if fakes:
        legend_lines.append("Fake victims: " + ", ".join(fakes))
    if lavas:
        legend_lines.append("Lava: " + ", ".join(lavas))

    return local_view, full_map, "\n".join(legend_lines)


def _to_relative_narrative(ax, ay, adir, ox, oy):
    """Translates global (ox, oy) to relative 'ahead/behind/left/right' for the LLM."""
    dx, dy = ox - ax, oy - ay

    # Rotate delta based on agent direction: 0:E, 1:S, 2:W, 3:N
    if adir == 0:  # East
        fwd, right = dx, dy
    elif adir == 1:  # South
        fwd, right = dy, -dx
    elif adir == 2:  # West
        fwd, right = -dx, -dy
    else:  # North
        fwd, right = -dy, dx

    parts = []
    if fwd > 0:
        parts.append("ahead")
    elif fwd < 0:
        parts.append("behind")

    if right > 0:
        parts.append("to the right")
    elif right < 0:
        parts.append("to the left")

    return " and ".join(parts) if parts else "at your position"


def to_semantic(obs: dict) -> str:
    """Narrative encoding: Includes walls and relative directions."""
    ax, ay = obs["agent_x"], obs["agent_y"]
    adir = obs["agent_dir"]
    grid = obs["grid"]
    x0, y0, x1, y1 = _cam_bounds(obs)
    carrying = obs["carrying"]

    in_fov, off_screen = [], []
    # Track wall directions to avoid listing every single wall tile
    wall_dirs = set()

    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            cell = int(cell)
            if cell == 0:
                continue  # Skip empty floor

            rel = _to_relative_narrative(ax, ay, adir, x, y)
            is_in_view = x0 <= x < x1 and y0 <= y < y1
            target = in_fov if is_in_view else off_screen

            if cell == 1:  # Wall
                if is_in_view:
                    wall_dirs.add(rel)
                continue

            if cell == LAVA:
                target.append(f"lava {rel}")
            elif cell == VICTIM:
                target.append(f"victim {rel}")
            elif cell == FAKE_VICTIM:
                target.append(f"fake victim {rel}")
            elif is_door(cell):
                d = decode_door(cell)
                state = (
                    "open"
                    if d["is_open"]
                    else ("locked" if d["is_locked"] else "closed")
                )
                target.append(f"{d['color']} door [{state}] {rel}")
            elif is_key(cell):
                target.append(f"{decode_key(cell)['color']} key {rel}")

    # Add summarized wall info to FOV
    if wall_dirs:
        in_fov.append(f"walls to your {', '.join(wall_dirs)}")

    # Immediate surroundings (1 step away)
    adx, ady, ldx, ldy = _DIR_VECTORS[adir]

    def _neighbor(dx, dy) -> str:
        nx, ny = ax + dx, ay + dy
        if not (0 <= nx < len(grid[0]) and 0 <= ny < len(grid)):
            return "out of bounds"
        cell = int(grid[ny][nx])
        if cell == 0:
            return "empty"
        if cell == 1:
            return "wall"
        if cell == LAVA:
            return "lava"
        if cell == VICTIM:
            return "victim"
        if is_door(cell):
            return "door"
        return "object"

    surroundings = (
        f"  Ahead: {_neighbor(adx, ady)}\n"
        f"  Left:  {_neighbor(ldx, ldy)}\n"
        f"  Right: {_neighbor(-ldx, -ldy)}\n"
        f"  Behind: {_neighbor(-adx, -ady)}"
    )

    return (
        f"AGENT STATUS:\n"
        f"- Facing: {_DIR_NAMES.get(adir, '?')}\n"
        f"- Inventory: {carrying.capitalize() + ' Key' if carrying else 'Empty'}\n"
        f"- Mission: {obs['saved_victims']} rescued, {obs['remaining_victims']} remaining.\n\n"
        f"IMMEDIATE SURROUNDINGS:\n{surroundings}\n\n"
        f"VISIBLE IN FOV (In camera view):\n- {', '.join(in_fov) if in_fov else 'Nothing actionable visible'}\n\n"
        f"OFF-SCREEN OBJECTS (Memory):\n- {', '.join(off_screen) if off_screen else 'None'}"
    )


def to_text(obs: dict) -> str:
    """Format the enriched obs dict as a human-readable summary for the LLM."""
    carrying = obs["carrying"]
    agent_char = _DIR_CHARS.get(obs["agent_dir"], "A")
    local_view, full_map, legend = _render(obs)
    return (
        f"Mission status: {obs['mission_status']}\n"
        f"Victims rescued: {obs['saved_victims']}\n"
        f"Victims remaining: {obs['remaining_victims']}\n"
        f"Steps taken: {obs['step_count']} / {obs['max_steps']}\n"
        f"Inventory: {carrying.capitalize() + ' Key' if carrying else 'None'}\n"
        f"Agent at ({obs['agent_x']},{obs['agent_y']}) facing {_DIR_NAMES.get(obs['agent_dir'], 'Unknown')}\n\n"
        f"Agent's local view (north=top, agent={agent_char} at center):\n"
        f"{local_view}\n\n"
        f"Full map (. empty  # wall  ~ lava  D door  K key  V victim  F fake  > v < ^ agent):\n"
        f"{full_map}\n\n"
        # f"{legend}"
    )


def build_prompt(obs: dict, prompt_type: str = "semantic") -> str:
    """Build a complete LLM system prompt for the given obs and prompt type."""
    template = _PROMPT_TEMPLATES.get(prompt_type, _PROMPT_TEMPLATES["detailed"])
    game_state = to_semantic(obs)
    return template.format(game_state=game_state)
