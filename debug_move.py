import sys
sys.path.insert(0, 'src')
from game.tutorial_env import TutorialEnv

env = TutorialEnv(start_part=1, screen_size=64, render_mode='rgb_array', tile_size=8)
env.reset()
print('agent_pos after reset ->', getattr(env, 'agent_pos', None))
forward = getattr(env.actions, 'forward', None)
left = getattr(env.actions, 'left', None)
print('forward:', forward, ' left:', left)
if forward is not None:
    env.step(forward)
    print('agent_pos after step forward ->', getattr(env, 'agent_pos', None))
else:
    print('No forward action available')
