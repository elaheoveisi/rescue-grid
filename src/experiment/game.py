from __future__ import annotations

import os
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

    Parameters
    ----------
    parameters : dict
        May include ``prompt_type`` ("sparse" | "detailed"),
        ``model`` (e.g. "gpt-4o-mini"), and ``provider`` ("openai" | "gemini").
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
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(parameters.get("display", 0))
        self.gui = SAREnvGUI(
            env,
            fullscreen=parameters.get("fullscreen", False),
            prompt_type=parameters.get("prompt_type", "detailed"),
            model=parameters.get("model", "gpt-4o-mini"),
            provider=parameters.get("provider", "openai"),
        )

    def initialize(self) -> None:
        self.gui.reset()

    def clean_up(self) -> None:
        pygame.quit()

    def read_data(self) -> list[str] | None:
        user = self.gui.user
        obs = user.obs
        state = {
            **(obs if isinstance(obs, dict) else {}),
            "action": user.last_action,
            "reward": user.last_reward,
            "terminated": user.terminated,
            "truncated": user.truncated,
            "prompt_type": user.prompt_type,
            "llm_model": user.model,
            "llm_provider": user.provider,
            "llm_response": user.last_llm_response,
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

    One block with 4 trials (sparse×openai, detailed×openai, sparse×gemini,
    detailed×gemini). Trial order is randomized at execution time.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        block = Block("sar_game_block")
        block.add_trial(
            SARGameTrial(
                "trial_sparse_openai",
                {
                    **config,
                    "prompt_type": "sparse",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                },
            ),
            order=1,
        )
        block.add_trial(
            SARGameTrial(
                "trial_detailed_openai",
                {
                    **config,
                    "prompt_type": "detailed",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                },
            ),
            order=2,
        )
        block.add_trial(
            SARGameTrial(
                "trial_sparse_gemini",
                {
                    **config,
                    "prompt_type": "sparse",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                },
            ),
            order=3,
        )
        block.add_trial(
            SARGameTrial(
                "trial_detailed_gemini",
                {
                    **config,
                    "prompt_type": "detailed",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                },
            ),
            order=4,
        )
        self.add_block(block)

    def get_data_signature(self) -> dict[str, Any]:
        return {
            "name": "SARGame",
            "type": "GameState",
            "channel_count": 1,
            "nominal_srate": 0.0,
            "channel_format": "string",
            "source_id": "rescue-grid-sar-game",
        }

    def execute(self, order: str = "random") -> list:
        super().execute(order)
        return []
