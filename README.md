<div align="center">

# 🚁 Rescue-Grid

### *Search. Navigate. Rescue.*

A Search and Rescue simulation environment where every second counts!

Built on top of [MiniGrid](https://github.com/Farama-Foundation/Minigrid) 🎮

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MiniGrid](https://img.shields.io/badge/built%20on-MiniGrid-green.svg)](https://github.com/Farama-Foundation/Minigrid)

</div>

---

## 🎯 What is Rescue-Grid?

Imagine a building on fire. Victims are trapped. Some rooms are locked. Lava (yes, lava!) blocks your path. Your mission? **Save everyone before time runs out.**

Rescue-Grid is a reinforcement learning environment that puts your agent in high-stakes rescue scenarios. Can your AI learn to be a hero?

## ✨ Features

| Feature                    | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| 🏢 **Multi-Room Layouts**   | Navigate through configurable grid-based buildings        |
| 🎯 **Real vs Fake Victims** | Distinguish cross-shaped victims ✚ from T-shaped decoys ⊤ |
| 🔥 **Lava Hazards**         | One wrong step and it's game over!                        |
| 🔐 **Locked Rooms**         | Find keys to unlock doors and reach trapped victims       |
| 🎮 **Interactive GUI**      | Beautiful Pygame interface with real-time info            |
| 🤖 **RL-Ready**             | Gymnasium compatible for training your rescue agents      |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Rescue-Grid.git
cd Rescue-Grid

# Install dependencies
pip install -r requirements.txt
pip install minigrid pygame pygame_gui pyyaml
```

### Run Your First Rescue Mission

```python
from game.sar.env import PickupVictimEnv
from game.sar.utils import VictimPlacer
from game.gui.main import SAREnvGUI

# Set up the mission
victim_placer = VictimPlacer(
    num_fake_victims=5,    # 5 decoys to fool you
    num_real_victims=3,    # 3 real victims to save
    important_victim="down"
)

# Create the environment
env = PickupVictimEnv(
    num_rows=3,
    num_cols=3,
    screen_size=800,
    render_mode="rgb_array",
    agent_pov=True,        # First-person view 👀
    add_lava=True,         # Danger mode: ON 🔥
    lava_per_room=2,
    locked_room_prob=0.5,  # 50% rooms are locked 🔐
    tile_size=64,
    victim_placer=victim_placer,
)

# Launch the mission!
env.reset()
gui = SAREnvGUI(env, fullscreen=False)
gui.run()
```

## ⚙️ Configuration

| Parameter          | Description              | Default |
| ------------------ | ------------------------ | ------- |
| `num_rows`         | Building height (rooms)  | 3       |
| `num_cols`         | Building width (rooms)   | 3       |
| `room_size`        | Tiles per room           | 8       |
| `add_lava`         | Enable lava hazards 🔥    | True    |
| `lava_per_room`    | Lava tiles per room      | 0       |
| `locked_room_prob` | Chance of locked doors 🔐 | 0.5     |
| `agent_pov`        | First-person view 👁️      | False   |

## 🎮 Controls

| Key     | Action                       |
| ------- | ---------------------------- |
| ⬆️       | Move forward                 |
| ⬅️ ➡️     | Rotate left/right            |
| `Space` | Toggle/interact (open doors) |
| `Tab`   | Pickup/rescue victim         |
| `Shift` | Drop object                  |
| `F11`   | Toggle fullscreen            |
| `ESC`   | Quit mission                 |

## 📁 Project Structure

```
src/
├── game/
│   ├── core/          # 🏗️ Base environment & level generation
│   ├── gui/           # 🖼️ Pygame GUI components
│   └── sar/           # 🚨 SAR-specific logic & objects
├── main.py            # 🚀 Entry point
└── utils.py           # 🔧 Utility functions
```

## 🤝 Contributing

Found a bug? Have an idea? PRs are welcome!

## 📄 License

MIT License - Feel free to use this for your research!

---

<div align="center">

**Built with ❤️ for the RL research community**

*Remember: In Rescue-Grid, heroes aren't born — they're trained!*

</div>
