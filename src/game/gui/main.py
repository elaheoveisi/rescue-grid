import numpy as np
import pygame
import pygame_gui

from .chat import ChatPanel
from .info import InfoPanel
from .user import User

pygame.init()


class SAREnvGUI:
    def __init__(self, env, config: dict | None = None):
        config = config or {}
        self.user = User(
            env,
            prompt_type=config.get("prompt_type", "detailed"),
            model=config.get("model", "gpt-4o-mini"),
            provider=config.get("provider", "openai"),
        )
        self.llm_nudge_interval = config.get("llm_nudge_interval", 10)
        self.fullscreen = config.get("fullscreen", False)
        self.game_size = self.user.env.screen_size

        self.panel_width = self.game_size // 2
        self.window_size = (self.game_size + self.panel_width, self.game_size)

        self.window = env.window

        if self.window is None:
            self._setup_display(display=config.get("display", 0))

        self._calculate_offsets()

        self.running = True
        self.clock = pygame.time.Clock()

        self._create_panels()

    def _setup_display(self, display=0):
        """Set up the pygame display window based on fullscreen state."""
        if self.fullscreen:
            self.window = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN, display=display
            )
            self.screen_size = self.window.get_size()
        else:
            self.window = pygame.display.set_mode(self.window_size)
            self.screen_size = self.window_size

    def _create_panels(self):
        """Create (or recreate) the UI manager and side panels."""
        self.manager = pygame_gui.UIManager(self.window_size, "src/game/gui/theme.json")
        self.info_panel = InfoPanel(self.manager, self.game_size, self.panel_width)
        self.chat_panel = ChatPanel(
            self.manager,
            self.game_size,
            self.panel_width,
            self.panel_width,
            self.panel_width,
        )

    def _calculate_offsets(self):
        """Calculate scale and offsets to center game content on screen."""
        scale_x = self.screen_size[0] / self.window_size[0]
        scale_y = self.screen_size[1] / self.window_size[1]
        self.scale = min(scale_x, scale_y)
        self.scaled_width = int(self.window_size[0] * self.scale)
        self.scaled_height = int(self.window_size[1] * self.scale)
        self.offset_x = (self.screen_size[0] - self.scaled_width) // 2
        self.offset_y = (self.screen_size[1] - self.scaled_height) // 2

    def render(self, frame):
        self.window.fill((0, 0, 0))

        combined_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)

        frame = np.transpose(frame, (1, 0, 2))
        game_surface = pygame.surfarray.make_surface(frame)
        game_surface = pygame.transform.smoothscale(
            game_surface, (self.game_size, self.game_size)
        )
        combined_surface.blit(game_surface, (0, 0))

        self.info_panel.render(self.user.env)

        time_delta = self.clock.tick(30) / 1000.0
        self.manager.update(time_delta)
        self.manager.draw_ui(combined_surface)

        if self.scale != 1.0:
            combined_surface = pygame.transform.smoothscale(
                combined_surface, (self.scaled_width, self.scaled_height)
            )
        self.window.blit(combined_surface, (self.offset_x, self.offset_y))
        pygame.display.update()

    def reset(self):
        self.user.reset()

    def handle_user_input(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.close()
        elif event.key == pygame.K_F11:
            self.toggle_fullscreen()
        elif event.key in (pygame.K_LALT, pygame.K_RALT):
            try:
                self.chat_panel.set_message("Agent", "thinking...", color="#888888")
                self.render(self.user.get_frame())
                reply = self.user.ask_llm()
                self.chat_panel.set_message("Agent", reply)
            except Exception as e:
                self.chat_panel.set_message("Error", str(e), color="#DC143C")
        else:
            event.key = pygame.key.name(int(event.key))
            self.user.handle_key(event)
            if (
                self.user.steps_since_last_llm > 0
                and self.user.steps_since_last_llm % self.llm_nudge_interval == 0
            ):
                self.chat_panel.set_message(
                    "Agent", "Please let me know if any guidance is needed."
                )

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        self._setup_display()
        self._calculate_offsets()
        self._create_panels()

    def close(self):
        self.running = False

    def run(self):
        self.reset()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    break
                self.manager.process_events(event)
                self.handle_user_input(event)

            if self.running:
                self.render(self.user.get_frame())

        pygame.quit()
