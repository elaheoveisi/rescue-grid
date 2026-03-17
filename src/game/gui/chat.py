import pygame
from pygame_gui.elements import UILabel, UIPanel, UITextBox

_NUDGE_TEXT = "Agent: Please let me know if any guidance is needed by pressing ALT."
_NUDGE_DURATION_MS = 3000


class ChatPanel:
    """Chat panel using pygame_gui built-in elements."""

    def __init__(self, manager, x_position, y_position, panel_width, panel_height):
        self.manager = manager
        self.panel_width = panel_width
        self.panel_height = panel_height
        self._nudge_time: int | None = None
        self._nudge_font = pygame.font.SysFont(None, 54)

        self.panel = UIPanel(
            relative_rect=pygame.Rect(x_position, y_position, panel_width, panel_height),
            manager=manager,
        )
        self.title = UILabel(
            relative_rect=pygame.Rect(0, 10, panel_width, 40),
            text="CHAT",
            manager=manager,
            container=self.panel,
            object_id="#title",
        )
        self.text_box = UITextBox(
            html_text="",
            relative_rect=pygame.Rect(0, 0, panel_width - 15, panel_height - 70),
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.title},
        )

    def set_message(self, sender, text, color="#6495ED"):
        self.text_box.set_text(f"<font color='{color}'><b>{sender}:</b> {text}</font>")

    def clear_message(self):
        self.text_box.set_text("")

    def nudge(self):
        self._nudge_time = pygame.time.get_ticks()

    def render_nudge(self, window, offset_x: int, offset_y: int, game_size: int):
        if self._nudge_time is None:
            return
        if pygame.time.get_ticks() - self._nudge_time >= _NUDGE_DURATION_MS:
            return
        dim = pygame.Surface((game_size, game_size))
        dim.set_alpha(220)
        dim.fill((0, 0, 0))
        window.blit(dim, (offset_x, offset_y))
        surf = self._nudge_font.render(_NUDGE_TEXT, True, (255, 255, 255))
        cx = offset_x + game_size // 2 - surf.get_width() // 2
        cy = offset_y + game_size // 2 - surf.get_height() // 2
        window.blit(surf, (cx, cy))

    def poll_llm(self, user):
        if user.llm_thread is None:
            return
        if user.llm_thread.is_alive():
            dots = "." * (int(pygame.time.get_ticks() / 400) % 3 + 1)
            self.set_message("Agent", f"thinking{dots}", color="#888888")
        elif user.llm_result is not None:
            kind, value = user.llm_result
            if kind == "reply":
                self.set_message("Agent", value)
            else:
                self.set_message("Error", value, color="#DC143C")
            user.llm_thread = None
            user.llm_result = None
