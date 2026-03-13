from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl

from ..llm.client import ask, ask_goals
from ..llm.parser import Goal


class User(ManualControl):
    def __init__(
        self,
        env,
        prompt_type: str = "detailed",
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ):
        self.env = env
        self.obs = None
        self.prompt_type = prompt_type
        self.model = model
        self.provider = provider
        self.last_llm_response: str | None = None
        self.current_goals: list[Goal] = []
        self.total_steps = 0
        self.episode_ended = False

    def step(self, action: Actions):
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = int(action)
        self.last_reward = float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.total_steps += 1
        if terminated:
            self.episode_ended = True
            self.reset()
        elif truncated:
            self.episode_ended = True
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
        """Ask the LLM for advice based on the current game state.

        When prompt_type is 'decompose', returns a numbered mission plan.
        Otherwise returns a single action string.
        """

        if self.obs is None:
            self.last_llm_response = "No observation available yet."
            return self.last_llm_response

        self.current_goals = ask_goals(
            self.obs, model=self.model, provider=self.provider
        )
        self.last_llm_response = "\n".join(
            f"{i + 1}. {g}" for i, g in enumerate(self.current_goals)
        ) or "No goals parsed."

        return self.last_llm_response
