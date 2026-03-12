import pygame
from pygame_gui.elements import UILabel, UIPanel, UITextBox


class ChatPanel:
    """Chat panel using pygame_gui built-in elements."""

    def __init__(self, manager, x_position, y_position, panel_width, panel_height):
        """Initialize the chat panel with pygame_gui elements."""
        self.manager = manager
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.messages = []

        # Create main panel
        self.panel = UIPanel(
            relative_rect=pygame.Rect(
                x_position, y_position, panel_width, panel_height
            ),
            manager=manager,
        )

        # Title
        self.title = UILabel(
            relative_rect=pygame.Rect(0, 10, panel_width, 40),
            text="CHAT",
            manager=manager,
            container=self.panel,
            object_id="#title",
        )

        # Scrollable message area — top anchored to the bottom of the title
        self.text_box = UITextBox(
            html_text="",
            relative_rect=pygame.Rect(0, 0, panel_width - 15, panel_height - 70),
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.title},
        )

    def add_message(self, sender, text, color="#DCDCDC"):
        """Append a message. color is a hex string, e.g. '#32CD32'."""
        fragment = f"<font color='{color}'><b>{sender}:</b> {text}</font>"
        self.messages.append(fragment)
        self._refresh()

    def _refresh(self):
        self.text_box.set_text("<br>".join(self.messages))

    def clear_messages(self):
        """Clear all messages from the chat."""
        self.messages.clear()
        self.text_box.set_text("")

    def render(self):
        """pygame_gui handles drawing via the manager."""
        pass
