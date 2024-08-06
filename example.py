import numpy as np
import sys
import os

from gym_match3.envs.match3_env import Match3Env


env = Match3Env(90)

print(f"Total size of the game state: {env.observation_space}")
print(f"Number of actions in this game: {env.action_space}")

_last_obs, infos = env.reset()
dones = False
action_space = infos["action_space"]

# Initialize Pygame

# Initialize the display with the initial state

action_space = infos['action_space']
obs, _ = env.reset()
print(obs[24])

_,_,_,_,_  = env.step(20)