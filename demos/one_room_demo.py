#!/usr/bin/env python3
"""Simple non-GUI demo that prints the layout for the three "one room" tutorial parts.

This script intentionally does not modify any `gui`, `core`, or `sar` source files.
It loads the local `rescue-grid/src` package by adjusting `sys.path`.
"""
from __future__ import annotations

import os
import sys


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    # Ensure local package imports resolve to rescue-grid/src
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from game.tutorial_env import TutorialEnv

    for part in range(1, 4):
        env = TutorialEnv(start_part=part)
        try:
            env.reset()
        except Exception:
            # Some LevelGen setups call gen_mission during reset; call directly as a fallback
            try:
                env.gen_mission()
            except Exception:
                pass

        # Count real victims and keys on the grid
        victims = env.get_all_victims()
        key_positions = []
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if obj is None:
                    continue
                t = getattr(obj, "type", None) or obj.__class__.__name__
                if str(t).lower().startswith("key") or "Key" in obj.__class__.__name__:
                    key_positions.append((x, y, obj.__class__.__name__))

        print(f"Tutorial part {part}")
        print(f"  Real victims: {len(victims)}")
        for i, v in enumerate(victims, 1):
            print(f"    victim#{i}: {v}")
        print(f"  Keys found: {len(key_positions)}")
        for kx, ky, kcls in key_positions:
            print(f"    {kcls} at ({kx},{ky})")
        print("---")


if __name__ == "__main__":
    main()
