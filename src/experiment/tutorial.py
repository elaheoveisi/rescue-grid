# mot_task.py
from __future__ import annotations

from typing import Any

import pygame
from ixp.task import Task

from game.gui.main import SAREnvGUI
from game.tutorial_env import TutorialEnv


class SARTutorial(Task):
    """
    Sustained Attention to Response Task tutorial containing a block of SARTTrials.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        screen_height = pygame.display.Info().current_h

        env = TutorialEnv(
            num_rows=1,
            num_cols=1,
            screen_size=screen_height,
            render_mode="rgb_array",
            agent_pov=True,
        )

        env.reset()
        self.gui = SAREnvGUI(env, fullscreen=True)

    def execute(self, order: str = "predefined"):
        try:
            results = []
            self.gui.run()
            return results
        finally:
            pass
