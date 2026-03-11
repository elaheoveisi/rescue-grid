from openai import OpenAI

from game.sar.observations import (
    EMPTY,
    FAKE_VICTIM,
    LAVA,
    VICTIM,
    WALL,
    decode_door,
    decode_key,
    is_door,
    is_key,
)

DUMMY = True  # Set to False to use the real OpenAI API


_client = None

_DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
_DIR_CHARS = {0: ">", 1: "v", 2: "<", 3: "^"}

_SYSTEM_PROMPT = """\
You are an AI assistant embedded in a Search and Rescue (SAR) simulation.
The agent navigates a grid world to rescue victims. You can see the full map.

Available actions: left, right, forward, pickup, drop, toggle (open/close doors), done.

Current game state:
{game_state}"""


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _cell_symbol(cell: int) -> str:
    if cell == EMPTY:
        return "."
    if cell == WALL:
        return "#"
    if cell == LAVA:
        return "~"
    if cell == VICTIM:
        return "V"
    if cell == FAKE_VICTIM:
        return "F"
    if is_door(cell):
        return "D"
    if is_key(cell):
        return "K"
    return "?"


def ascii_map(obs: dict) -> str:
    """Render ASCII map with agent direction arrow."""
    ax, ay = obs["agent_x"], obs["agent_y"]
    agent_char = _DIR_CHARS.get(obs["agent_dir"], "A")
    lines = []
    for y, row in enumerate(obs["grid"]):
        row_str = ""
        for x, cell in enumerate(row):
            row_str += agent_char if (x == ax and y == ay) else _cell_symbol(int(cell))
        lines.append(row_str)
    return "\n".join(lines)


def object_legend(obs: dict) -> str:
    """List all doors, keys, victims, and lava with full color/state detail."""
    victims, fakes, lavas, doors, keys = [], [], [], [], []
    for y, row in enumerate(obs["grid"]):
        for x, cell in enumerate(row):
            cell = int(cell)
            if cell == LAVA:
                lavas.append(f"({x},{y})")
            elif cell == VICTIM:
                victims.append(f"({x},{y})")
            elif cell == FAKE_VICTIM:
                fakes.append(f"({x},{y})")
            elif is_door(cell):
                d = decode_door(cell)
                state = (
                    "open"
                    if d["is_open"]
                    else ("locked" if d["is_locked"] else "closed")
                )
                doors.append(f"  D at ({x},{y}) = {d['color']} [{state}]")
            elif is_key(cell):
                k = decode_key(cell)
                keys.append(f"  K at ({x},{y}) = {k['color']} key")

    lines = []
    if doors:
        lines.append("Doors:")
        lines.extend(doors)
    if keys:
        lines.append("Keys:")
        lines.extend(keys)
    if victims:
        lines.append("Victims at: " + ", ".join(victims))
    if fakes:
        lines.append("Fake victims at: " + ", ".join(fakes))
    if lavas:
        lines.append("Lava at: " + ", ".join(lavas))
    return "\n".join(lines)


def to_text(obs: dict) -> str:
    """Format the enriched obs dict as a human-readable summary for the LLM."""
    carrying = obs["carrying"]
    inventory = f"{carrying.capitalize()} Key" if carrying else "None"
    dir_name = _DIR_NAMES.get(obs["agent_dir"], "Unknown")
    return (
        f"Mission status: {obs['mission_status']}\n"
        f"Victims rescued: {obs['saved_victims']}\n"
        f"Victims remaining: {obs['remaining_victims']}\n"
        f"Steps taken: {obs['step_count']} / {obs['max_steps']}\n"
        f"Inventory: {inventory}\n"
        f"Agent at ({obs['agent_x']},{obs['agent_y']}) facing {dir_name}\n\n"
        f"Map (. empty  # wall  ~ lava  D door  K key  V victim  F fake  > v < ^ agent):\n"
        f"{ascii_map(obs)}\n\n"
        f"{object_legend(obs)}"
    )


def ask(obs, model: str = "gpt-4o-mini") -> str:
    """Synchronously ask the LLM for advice given the current game observation."""
    if DUMMY:
        return "Game state received. Keep searching for victims and avoid lava!"

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(game_state=to_text(obs)),
            },
            {
                "role": "user",
                "content": "What should I do next?",
            },
        ],
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()
