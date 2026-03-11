from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl

from .llm import ask, to_text


class User(ManualControl):
    def __init__(self, env):
        self.env = env
        self.obs = None

    def step(self, action: Actions):
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = int(action)
        self.last_reward = float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        print(to_text(self.obs))

        if terminated:
            print("terminated!")
            self.reset()
        elif truncated:
            print("truncated!")
            self.reset()
        else:
            self.env.render()

    def handle_key(self, event):
        key: str = event.key

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)

    def get_frame(self):
        return self.env.render()

    def reset(self):
        obs, info = self.env.reset()
        self.last_action = None
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.obs = obs

    def ask_llm(self) -> str:
        """Ask the LLM for advice based on the current game state."""

        return ask(self.obs)
