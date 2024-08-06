import numpy as np
import sys
import os

from gym_match3.envs.match3_env import Match3Env
from display.pygame_display import *

env = Match3Env(90)

print(f"Total size of the game state: {env.observation_space}")
print(f"Number of actions in this game: {env.action_space}")

_last_obs, infos = env.reset()
dones = False
action_space = infos["action_space"]

# Initialize Pygame
pygame.init()

# Initialize the display with the initial state


def action_to_coords(action):
    if action < 80:
        row = action//8
        col = action%8
        coord1 = (row, col)
        coord2 = (row, col+1)
    else:
        action = action - 80
        row = action//9
        col = action%9
        coord1 = (row, col)
        coord2 = (row+1,col)

    return coord1, coord2


matrix = np.array(env.return_game_matrix)
display = Display(matrix)
for i in range(100):
    # Identify the indices where the value is 1
    action_space = infos['action_space']

    indices_with_one = [index for index, value in enumerate(action_space) if value == 1]

    # Randomly select one of those indices
    if indices_with_one:
        # Temporary model instantiation (replace with actual model loading if necessary)
        # model = DQN(1, 161).to(DEVICE)

        # input_tensor = torch.tensor(env.return_game_matrix, dtype=torch.float).to(DEVICE)
        
        # output_tensor = model(input_tensor)

        # print("outpute tensore", output_tensor)
        # output_tensor[0] = 100
        # print('new output tensor', output_tensor)
        # print("output tensor type", type(output_tensor))

        # max_val, max_idx = torch.max(output_tensor, dim=0)

        # in the future I will let the model choose the selected action here
        # selected_action = int(input("Put the move you want to do on the board: "))
        selected_action = random.choice(indices_with_one)


        # obs is a 26,10,9 tensor, with each layer representing an observation (e.g layer of greens, layer of blues, layer of monter, layer of each powerup)
        # reward is a dictionary with score, cancel_score, creapt_pu_score, match_damage_on_monster, power_damage_on_monster, and damage_on_user keys
        # episode over just signifies whether or not the player/creep has died
        # infos is just the dictionary containing the action space of moves that would result in a match
        obs, reward, episode_over, infos = env.step(selected_action)
        print(obs['legal_action'])
        # print("1", obs[1])
        # print("2", obs[2])
        # print("3:", obs[3])
        # print("4:", obs[4])
        # print("5:", obs[5])
        # print("monster:", obs[13])
        # print('legal_action:', obs[24])
        # print(action_space)
        # print('dones', dones)
        # print("infos: ", infos)

        # display and update matrix
        (row1,col1), (row2,col2) = action_to_coords(selected_action)
        display.animate_switch((row1,col1),(row2,col2), matrix)
        matrix = np.array(env.return_game_matrix)
        display.update_display(matrix)

        print("reward: ", reward)


    pygame.time.wait(1000)  # Small delay to make the loop more manageable

pygame.quit()
sys.exit()

#     else:
#         print("No indices with value 1 found.")
#         dones = True

# getting the model to predict the values for the game

# make the training function for the model
    # find out how we can input the states into the training thing
    # configure the training function to work for one instance of the inputted state
    # how am I going to configure the reward as well?
        # come back to this when you get there 