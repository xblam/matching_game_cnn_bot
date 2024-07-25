import numpy as np
from model import *


from gym_match3.envs.match3_env import Match3Env
from display.display import display_matrix
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
matrix = np.array(env.return_state)
display = Display(matrix)

def action_to_coords(action):
    coords_to_switch = []
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

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for i in range(100):
        # Identify the indices where the value is 1
        indices_with_one = [index for index, value in enumerate(action_space) if value == 1]

        # Randomly select one of those indices
        if indices_with_one:
            # Temporary model instantiation (replace with actual model loading if necessary)
            model = DQN(1, 161).to(DEVICE)

            input_tensor = torch.tensor(env.return_state, dtype=torch.float).to(DEVICE)
            output_tensor = model(input_tensor)

            max_val, max_idx = torch.max(output_tensor, dim=0)
            # selected_action = int(input("Put the move you want to do on the board: "))
            selected_action = np.random.randint(0,161)

            obs, reward, dones, infos = env.step(int(selected_action))

            print("obs:", obs)
            print("obs[0]:", obs[0])
            print("obs shape:", obs.shape)
            print("reward:", reward)
            print("dones:", dones)
            print("infos:", infos)
            (x1,y1), (x2,y2) = action_to_coords(selected_action)
            display.animate_switch((x1,y1),(x2,y2))
            matrix = np.array(env.return_state)

            display.update_display(matrix)

            print("Selected index:", selected_action)
            print("Reward of this action:", reward)

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