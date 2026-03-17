from pathlib import Path

import pygame
from pygame_gui.elements import UIImage, UILabel, UIPanel, UITextBox


class InfoPanel:
    """Info panel using pygame_gui built-in elements."""

    def __init__(self, manager, x_offset, panel_width, panel_height=None):
        """Initialize the info panel with pygame_gui elements."""
        self.manager = manager
        self.panel_x = x_offset
        self.panel_width = panel_width
        panel_height = panel_height if panel_height is not None else panel_width

        # Create main panel
        self.panel = UIPanel(
            relative_rect=pygame.Rect(self.panel_x, 0, panel_width, panel_height),
            manager=manager,
        )

        # Layout constants
        PADDING_X = 10
        content_width = panel_width - (PADDING_X * 2)

        # Title
        self.title = UILabel(
            relative_rect=pygame.Rect(0, 1, panel_width, 40),
            text="MISSION INFORMATION",
            manager=manager,
            container=self.panel,
            object_id="#title",
        )

        # Trial name — created but hidden; call set_trial_name() to update
        self.trial_label = UILabel(
            relative_rect=pygame.Rect(0, 1, panel_width, 25),
            text="",
            manager=manager,
            container=self.panel,
            object_id="#info_text",
            anchors={"top": "top", "top_target": self.title},
        )
        self.trial_label.hide()

        # Victims section
        self.victims_header = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="VICTIMS",
            manager=manager,
            container=self.panel,
            object_id="#section_header",
            anchors={"top": "top", "top_target": self.title},
        )

        self.rescued_label = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="Rescued: 0",
            manager=manager,
            container=self.panel,
            object_id="label",
            anchors={"top": "top", "top_target": self.victims_header},
        )

        self.remaining_label = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="Remaining: 0",
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.rescued_label},
        )

        self.score_label = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="Score: 0",
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.remaining_label},
        )

        # Time & Inventory section
        self.time_header = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="TIME & INVENTORY",
            manager=manager,
            container=self.panel,
            object_id="#section_header",
            anchors={"top": "top", "top_target": self.score_label},
        )

        self.steps_label = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="Steps: 0 / 0",
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.time_header},
        )

        self.inventory_label = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="Inventory: None",
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.steps_label},
        )

        # Navigation section
        self.nav_header = UILabel(
            relative_rect=pygame.Rect(PADDING_X, 1, content_width, 24),
            text="NAVIGATION",
            manager=manager,
            container=self.panel,
            object_id="#section_header",
            anchors={"top": "top", "top_target": self.inventory_label},
        )

        compass_size = panel_width // 3
        raw = pygame.image.load(
            str(Path(__file__).parent / "compass.png")
        ).convert_alpha()
        rgb = pygame.surfarray.pixels3d(raw)
        rgb[:] = 255 - rgb
        del rgb
        compass_surface = pygame.transform.smoothscale(
            raw, (compass_size, compass_size)
        )
        self.compass_image = UIImage(
            relative_rect=pygame.Rect(PADDING_X, 1, compass_size, compass_size),
            image_surface=compass_surface,
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.nav_header},
        )

        # Controls beside the compass
        controls_x = PADDING_X + compass_size + 50
        controls_width = panel_width - controls_x - 10
        controls_html = "Arrows &rarr; Move / Turn<br>Tab &rarr; Pick up<br>Shift &rarr; Drop Item<br>Space &rarr; Open door<br>Alt &rarr; Ask Guidance"
        UITextBox(
            html_text=controls_html,
            relative_rect=pygame.Rect(controls_x, 1, controls_width, compass_size + 10),
            manager=manager,
            container=self.panel,
            object_id="#control_text",
            anchors={"top": "top", "top_target": self.nav_header},
        )

        # Status message (anchored to bottom)
        self.status_label = UILabel(
            relative_rect=pygame.Rect(PADDING_X, -50, content_width, 40),
            text="",
            manager=manager,
            container=self.panel,
            object_id="#success_text",
            anchors={"bottom": "bottom"},
        )
        self.status_label.hide()

    def set_trial_name(self, name: str):
        self.trial_label.set_text(name)

    def _update_victims_section(self, mission_status):
        """Update the victims section labels."""
        rescued = mission_status.get("saved_victims", 0)
        remaining = mission_status.get("remaining_victims", 0)

        self.rescued_label.set_text(f"Rescued: {rescued}")
        self.remaining_label.set_text(f"Remaining: {remaining}")
        self.score_label.set_text(f"Score: {rescued * 10}")

        # Update remaining color based on count
        if remaining > 0:
            self.remaining_label.change_object_id("#danger_text")
        else:
            self.remaining_label.change_object_id("#success_text")

    def _update_time_and_inventory(self, env):
        """Update time and inventory labels."""
        steps = getattr(env, "step_count", 0)
        max_steps = getattr(env, "max_steps", 0)
        carrying = getattr(env, "carrying", None)
        self.steps_label.set_text(f"Steps: {steps} / {max_steps}")

        if carrying:
            inventory_text = (
                f"Inventory: {carrying.color.capitalize()} {carrying.type.capitalize()}"
            )
            # Set color based on key color
            key_color_map = {
                "red": "#red_key",
                "green": "#green_key",
                "blue": "#blue_key",
                "yellow": "#yellow_key",
                "purple": "#purple_key",
                "grey": "#grey_key",
            }
            color_id = key_color_map.get(carrying.color.lower(), "#info_text")
            self.inventory_label.change_object_id(color_id)
            self.inventory_label.set_text(inventory_text)
        else:
            self.inventory_label.change_object_id("label")
            self.inventory_label.set_text("Inventory: None")

    def _update_status(self, mission_status):
        """Update the status message label."""
        status = mission_status.get("status", "incomplete")
        if status == "success":
            self.status_label.change_object_id("#success_text")
            self.status_label.set_text("MISSION COMPLETE!")
        elif status == "failure":
            self.status_label.change_object_id("#danger_text")
            self.status_label.set_text("MISSION FAILED")
        else:
            self.status_label.set_text("")

    def render(self, obs, env):
        """Update the panel with current game state."""
        mission_status = {
            "status": obs.get("mission_status", "incomplete"),
            "saved_victims": obs.get("saved_victims", 0),
            "remaining_victims": obs.get("remaining_victims", 0),
        }
        self._update_victims_section(mission_status)
        self._update_time_and_inventory(env)
        self._update_status(mission_status)
