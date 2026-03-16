import pygame
from pygame_gui.elements import UILabel, UIPanel, UITextBox


class ChatPanel:
    """Chat panel using pygame_gui built-in elements."""

    def __init__(self, manager, x_position, y_position, panel_width, panel_height):
        """Initialize the chat panel with pygame_gui elements."""
        self.manager = manager
        self.panel_width = panel_width
        self.panel_height = panel_height

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

    def set_message(self, sender, text, color="#6495ED"):
        """Display a message in the chat, replacing any previous one."""
        self.text_box.set_text(f"<font color='{color}'><b>{sender}:</b> {text}</font>")

    def clear_message(self):
        """Clear the chat display."""
        self.text_box.set_text("")
