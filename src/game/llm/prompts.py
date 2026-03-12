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

_DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
_DIR_CHARS = {0: ">", 1: "v", 2: "<", 3: "^"}


# --- Sparse prompt: LLM replies with one or two action words only ---
SPARSE_SYSTEM_PROMPT = """\
You are the Tactical Lead for a Search and Rescue (SAR) operation.
You have access to game state given below. Study the map in detail, understand other info, and issue a command to help the user complete the mission.

TREE OF THOUGHT REASONING:
1. Inventory Check: Confirm the agent is not already carrying a key before a 'pickup' for next 'key'.
2. Path Planning: Account for the fact that the agent cannot move backward; it must turn to change direction.
3. Prioritization: Determine if a victim can be reached directly or if a door must be unlocked first.

EXAMPLE COMMAND GRAMMAR:
- "pick up the [color] [key"
- "drop the [color] [key]"
- "open the [color] door"
- "pick up [victim] in the room towards left"

SAR OBJECTIVES and RULES:
1. Priority 1: Locate and retrieve as many real victimes as possible.
2. Priority 2: Clear obstacles (Open/Unlock Doors). Avoid Lava.
2. Rule: You can carry ONLY one 'key' at a time. If holding a key, you must 'drop' it to 'pickup' another 'key'.
3. Rule: No "Move Back" action exists.

DO NOT:
1. Use numbers like "go to key at (4, 5)" or "pick up victim at (4, 5)"

OUTPUT:
Reply with ONLY the mission string. Do NOT use numbers, bullet points, or punctuation.
Example: drop the blue key or pick up the victim

Current game state:
{game_state}"""

# --- Detailed prompt: LLM replies with 1–2 sentence guidance ---
DETAILED_SYSTEM_PROMPT = """\
You are the Tactical Lead for a Search and Rescue (SAR) operation.
You have access to game state given below. Study the map in detail, understand other info, and issue a command to help the user complete the mission.

TREE OF THOUGHT REASONING:
1. Analyze the agent's current position, orientation, and inventory.
2. Compare 2-3 possible next steps (e.g., "Find the key" vs. "Explore the below room").
3. Evaluate which step maximizes victim safety and minimizes total moves.

EXAMPLE COMMAND GRAMMAR:
- Forward, Left, Right, Pickup, Drop, Toggle (for doors).

SAR OBJECTIVES and RULES:
1. Priority 1: Locate and retrieve as many real victimes as possible.
2. Priority 2: Clear obstacles (Open/Unlock Doors). Avoid Lava.
2. Rule: You can carry ONLY one 'key' at a time. If holding a key, you must 'drop' it to 'pickup' another 'key'.
3. Rule: No "Move Back" action exists.

DO NOT:
1. Use numbers like "go to key at (4, 5)" or "pick up victim at (4, 5)"

OUTPUT:
Provide 1–2 sentences of guidance. Reply with ONLY the mission message. Do NOT use numbers.
Structure: [Action Verb] -> [Target Object] -> [Reason].
Example: "Turn left and move forward to reach the blue key; you need it to unlock the room where the victim is trapped."

Current game state:
{game_state}"""

# Default prompt (kept for backwards compatibility)
SYSTEM_PROMPT = DETAILED_SYSTEM_PROMPT


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


def sparse_prompt(obs: dict) -> str:
    """Build a sparse prompt — LLM should reply with action word(s) only."""
    return SPARSE_SYSTEM_PROMPT.format(game_state=to_text(obs))


def detailed_prompt(obs: dict) -> str:
    """Build a detailed prompt — LLM should reply with 1–2 sentence guidance."""
    return DETAILED_SYSTEM_PROMPT.format(game_state=to_text(obs))


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
