# mot_task.py
from __future__ import annotations

from typing import Any
import os
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
        self.gui: SAREnvGUI | None = None

    def _build_gui(self) -> SAREnvGUI:
        screen_height = pygame.display.Info().current_h

        env = TutorialEnv(
            num_rows=1,
            num_cols=1,
            screen_size=screen_height,
            render_mode="rgb_array",
            agent_pov=True,
        )

        env.reset()
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(
            self.config.get("display", 0)
        )
        return SAREnvGUI(
            env,
            fullscreen=self.config.get("fullscreen", False),
            prompt_type=self.config.get("prompt_type", "detailed"),
            model=self.config.get("model", "gpt-4o-mini"),
            provider=self.config.get("provider", "openai"),
            display=self.config.get("display", 0),
        )
        
    def initialize(self) -> None:
        if self.gui is None:
            self.gui = self._build_gui()
        self.gui.reset()

    def clean_up(self) -> None:
        self.gui = None
        pygame.quit()

    def execute(self, order: str = "predefined"):
        self.initialize()
        try:
            results = []
            if self.gui is not None:
                self.gui.run()
            return results
        finally:
            self.clean_up()
