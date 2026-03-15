import pygame
import yaml
from ixp.sensors.eye_tracker.tobii import TobiiEyeTracker

from game.gui.main import SAREnvGUI
from game.sar.env import PickupVictimEnv
from game.sar.utils import VictimPlacer

try:
    from game.tutorial_env import TutorialEnv
except ImportError:
    pass
from utils import skip_run

# Load config
config_path = "configs/config.yml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


with skip_run("skip", "sar_gui_advanced") as check, check():
    # Access the width and height of the current display
    screen_height = pygame.display.Info().current_h
    victim_placer = VictimPlacer(
        num_fake_victims=5, num_real_victims=3, important_victim="down"
    )
    env = PickupVictimEnv(
        num_rows=3,
        num_cols=3,
        screen_size=800,
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
    gui = SAREnvGUI(env, fullscreen=False)
    gui.run()


with skip_run("skip", "tutorial") as check, check():
    # Access the width and height of the current display
    screen_height = pygame.display.Info().current_h

    env = TutorialEnv(
        num_rows=1,
        num_cols=1,
        screen_size=800,
        render_mode="rgb_array",
        agent_pov=True,
    )

    env.reset()
    gui = SAREnvGUI(env, fullscreen=True)
    gui.run()


with skip_run("skip", "tobii") as check, check():
    tobii = TobiiEyeTracker()
    tobii.initialize()
    tobii.calibrate()


with skip_run("run", "tobii") as check, check():
    from dotenv import load_dotenv

    load_dotenv()
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.google_genai import GoogleGenAI

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="Tell me a story"),
    ]
    llm = GoogleGenAI(model="gemini-2.5-flash")
    resp = llm.chat(messages)

    print(resp)
