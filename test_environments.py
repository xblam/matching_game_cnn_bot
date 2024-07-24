import numpy as np
from model import *


from gym_match3.envs.match3_env import Match3Env
from display.display import display_matrix

env = Match3Env(90)

print(f"Total size of the game state: {env.observation_space}")
print(f"Number of actions in this game: {env.action_space}")

_last_obs, infos = env.reset()
dones = False
action_space = infos["action_space"]

for i in range(100):
    # Identify the indices where the value is 1
    indices_with_one = [index for index, value in enumerate(action_space) if value == 1]
    # Randomly select one of those indices
    if indices_with_one:
        
        # this is just here temporarily BUT MAKE SURE THE MODEL GETS CHANGED AFTER EVERY MOVE OR ELSE IF IT OUTPUTS AN ILLEGAL MOVE YOU WILL BE STUCK FOREVER
        model = DQN(1, 161).to(DEVICE)
        # selected_action = random.choice(indices_with_one)
        input_tensor = torch.tensor(env.return_state, dtype=torch.float).to(DEVICE)

        output_tensor = model(input_tensor)

        max_val, max_idx = torch.max(output_tensor, dim = 0)

        old_matrix = np.array(env.return_state)

        selected_action = int(input("put the move you want to do on the board: "))
        obs, reward, dones, infos = env.step(int(selected_action))
        print("obs: ", obs)
        
        print("obs: ", obs[0])
        print("obs shape: ", obs.shape)
        print("reward: ", reward)
        print("dones: ", dones)
        print("infos: ", infos)
 
 
 
        print("Selected index:", selected_action)
        print("Reward of this action:", reward)
 
        matrix = np.array(env.return_state)
        display_matrix(old_matrix)
        display_matrix(matrix)


    else:
        print("No indices with value 1 found.")
        dones = True

# getting the model to predict the values for the game

# make the training function for the model
    # find out how we can input the states into the training thing
    # configure the training function to work for one instance of the inputted state
    # how am I going to configure the reward as well?
        # come back to this when you get there 