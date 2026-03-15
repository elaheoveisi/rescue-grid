"""BFS-based pathfinding utilities for LLM prompt generation.

All queries operate on the encoded integer grid from obs["grid"].
Lava and walls are impassable. Locked doors are impassable unless the
agent is carrying the matching colour key.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from game.sar.observations import (
    LAVA,
    WALL,
    VICTIM,
    FAKE_VICTIM,
    decode_door,
    decode_key,
    is_door,
    is_key,
)

_EMPTY = 0


@dataclass
class PathResult:
    reachable: bool
    path_length: int | None  # steps to stand adjacent and interact
    blocking_doors: list[str]  # colours of locked doors on shortest path


def _is_passable(cell: int, carrying: str | None) -> bool:
    """Return True if the agent can move through this cell."""
    if cell == _EMPTY:
        return True
    if cell == WALL or cell == LAVA:
        return False
    if is_door(cell):
        d = decode_door(cell)
        if d["is_open"]:
            return True
        if d["is_locked"]:
            return carrying is not None and d["color"] == carrying
        return True  # closed but unlocked — agent can push through
    if is_key(cell):
        return True  # keys don't block movement
    # victims and fake victims block movement (can_overlap = False)
    return False


def bfs(
    grid: list[list[int]],
    start: tuple[int, int],
    target: tuple[int, int],
    carrying: str | None,
) -> PathResult:
    """BFS from start to any cell adjacent to target.

    Returns a PathResult with reachability, step count, and any locked
    doors that appear on the shortest path.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    tx, ty = target

    # Neighbours of target that the agent can stand on to interact
    interaction_cells = [
        (tx + dx, ty + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if 0 <= tx + dx < cols and 0 <= ty + dy < rows
    ]

    if not interaction_cells:
        return PathResult(reachable=False, path_length=None, blocking_doors=[])

    # BFS: state = (x, y), track path length and doors encountered
    # We store (x, y, frozenset_of_locked_doors_seen) but that's expensive.
    # Instead just do plain BFS for path length; collect locked doors
    # on the winning path via parent tracking.

    visited: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        cx, cy = queue.popleft()

        if (cx, cy) in interaction_cells:
            # Reconstruct path to collect blocking doors
            path = []
            node: tuple[int, int] | None = (cx, cy)
            while node is not None:
                path.append(node)
                node = visited[node]
            path.reverse()

            blocking = []
            for px, py in path:
                cell = int(grid[py][px])
                if is_door(cell):
                    d = decode_door(cell)
                    if d["is_locked"]:
                        blocking.append(d["color"])

            return PathResult(
                reachable=True,
                path_length=len(path) - 1,  # steps taken
                blocking_doors=blocking,
            )

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if (nx, ny) in visited:
                continue
            cell = int(grid[ny][nx])
            if not _is_passable(cell, carrying):
                continue
            visited[(nx, ny)] = (cx, cy)
            queue.append((nx, ny))

    return PathResult(reachable=False, path_length=None, blocking_doors=[])


def query_all_objects(obs: dict) -> dict[tuple[int, int], PathResult]:
    """Run BFS from the agent to every non-empty, non-wall cell in the grid.

    Returns a mapping of (x, y) → PathResult for every object of interest.
    """
    grid = obs["grid"]
    ax, ay = obs["agent_x"], obs["agent_y"]
    carrying = obs.get("carrying")
    start = (ax, ay)

    results: dict[tuple[int, int], PathResult] = {}
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            cell = int(cell)
            if cell in (_EMPTY, WALL):
                continue
            if (x, y) == start:
                continue
            results[(x, y)] = bfs(grid, start, (x, y), carrying)

    return results
