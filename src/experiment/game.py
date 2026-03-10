# mot_task.py
from __future__ import annotations


from typing import Any

import pygame

from game.gui.main import SAREnvGUI
from game.sar.env import PickupVictimEnv
from game.sar.utils import VictimPlacer

from ixp.task import Task



class SARGame(Task):
    """
    Sustained Attention to Response Task tutorial containing a block of SARTTrials.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        screen_height = pygame.display.Info().current_h
        victim_placer = VictimPlacer(
            num_fake_victims=5, num_real_victims=3, important_victim="down"
        )
        env = PickupVictimEnv(
            num_rows=3,
            num_cols=3,
            screen_size=screen_height,
            render_mode="rgb_array",
            agent_pov=True,
            add_lava=True,
            lava_per_room=2,
            locked_room_prob=0.5,
            # camera_strategy=FullviewCamera(),
            tile_size=64,
            victim_placer=victim_placer,
        )
        env.reset()
        self.gui = SAREnvGUI(env, fullscreen=False)
        


    def execute(self, order: str = 'predefined'):        
        try:
            results = []
            self.gui.run()
            return results
        finally:
            pass