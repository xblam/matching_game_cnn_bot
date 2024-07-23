import random
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np


from gym_match3.envs.match3_env import Match3Env
from display.display import display_matrix

env = Match3Env(90)

print(f"Total size of the game state{env.observation_space}")
print(f"Number of actions in this game{env.action_space}")

_last_obs, infos = env.reset()
dones = False
action_space = infos["action_space"]

for i in range(100):
    # Identify the indices where the value is 1
    indices_with_one = [index for index, value in enumerate(action_space) if value == 1]
    # Randomly select one of those indices
    if indices_with_one:

        selected_action = random.choice(indices_with_one)
        old_matrix = np.array(env.return_board())
        # selected_action = int(input("put the move you want to do on the board: "))
        print("Selected index:", selected_action)

        obs, reward, dones, infos = env.step(selected_action)
        # not really sure what infos means

        print("Reward of this action:", reward)

        matrix = np.array(env.return_board())
        display_matrix(old_matrix)
        display_matrix(matrix)


    else:
        print("No indices with value 1 found.")
        dones = True
