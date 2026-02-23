from minigrid.manual_control import ManualControl

from . import llm


class User(ManualControl):
    def __init__(self, env):
        self.env = env
        self.controller = ManualControl(env)

    def handle_key(self, event):
        self.controller.key_handler(event)

    def get_frame(self):
        return self.env.render()

    def reset(self):
        self.controller.reset()

    def get_game_context(self) -> str:
        """Format current env state as a text summary for the LLM."""
        status = self.env.get_mission_status()
        steps = getattr(self.env, "step_count", 0)
        max_steps = getattr(self.env, "max_steps", 0)
        carrying = getattr(self.env, "carrying", None)
        inventory = f"{carrying.color.capitalize()} Key" if carrying else "None"
        return (
            f"Mission status: {status['status']}\n"
            f"Victims rescued: {status['saved_victims']}\n"
            f"Victims remaining: {status['remaining_victims']}\n"
            f"Steps taken: {steps} / {max_steps}\n"
            f"Inventory: {inventory}"
        )

    def ask_llm(self) -> str:
        """Ask the LLM for advice based on the current game state."""
        return llm.ask(self.get_game_context())
