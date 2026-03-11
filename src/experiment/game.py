from __future__ import annotations

from typing import Any

import pygame
import ujson
from ixp.task import Block, LSLTrial, Task

from game.gui.main import SAREnvGUI
from game.sar.env import PickupVictimEnv
from game.sar.utils import VictimPlacer


class SARGameTrial(LSLTrial):
    """LSL-streaming trial for the SAR rescue game.

    Streams game state as a single JSON channel at each rendered frame.
    Captures enough data to fully replay or reconstruct the session:
    env config, per-step action/reward, agent state, and full object states.

    Block.execute() calls initialize() → execute() → clean_up(), so the
    full lifecycle is managed externally by the Block.
    """

    def __init__(self, trial_id: str, parameters: dict[str, Any]):
        super().__init__(trial_id, parameters)
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
            tile_size=64,
            victim_placer=victim_placer,
        )
        env.reset()
        self.gui = SAREnvGUI(env, fullscreen=False)
        self.create_lsl_stream()

    def initialize(self) -> None:
        self.gui.reset()

    def clean_up(self) -> None:
        pygame.quit()

    def get_data_signature(self) -> dict[str, Any]:
        return {
            "name": "SARGame",
            "type": "GameState",
            "channel_count": 1,
            "nominal_srate": 0.0,
            "channel_format": "string",
            "source_id": "rescue-grid-sar-game",
        }

    def read_data(self) -> list[str] | None:
        user = self.gui.user
        obs = user.obs
        state = {
            **(obs if isinstance(obs, dict) else {}),
            "action": user.last_action,
            "reward": user.last_reward,
            "terminated": user.terminated,
            "truncated": user.truncated,
        }
        return [ujson.dumps(state)]

    def execute(self) -> None:

        while self.gui.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.gui.close()
                    break
                self.gui.manager.process_events(event)
                self.gui.handle_user_input(event)

            if self.gui.running:
                self.gui.render(self.gui.user.get_frame())
                self.stream()


class SARGame(Task):
    """SAR game task for the ixp experiment framework.

    Wraps SARGameTrial in a Block so Experiment.add_task() / Experiment.run()
    can manage the full trial lifecycle including LSL streaming verification.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        block = Block("sar_game_block")
        block.add_trial(
            SARGameTrial(trial_id="sar_game_trial", parameters=config or {}),
            order=1,
        )
        self.add_block(block)

    def execute(self, order: str = "predefined") -> list:
        super().execute(order)
        return []
