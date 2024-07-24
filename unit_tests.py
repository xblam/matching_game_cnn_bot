
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model import *
from gym_match3.envs.match3_env import Match3Env

class TestBoard(unittest.TestCase):
    def model_3_layer_matrix(self):
       
       
        matrix_3_channels = np.array(([
        [14, 14, 2, 4, 3, 1, 4, 2, 4],
        [14, 14, 4, 3, 1, 2, 1, 3, 3],
        [3, 4, 1, 5, 2, 4, 1, 2, 5],
        [5, 5, 4, 5, 2, 5, 5, 4, 4],
        [4, 1, 2, 3, 1, 2, 3, 4, 2],
        [4, 1, 4, 4, 2, 4, 1, 3, 4],
        [2, 4, 3, 3, 5, 5, 4, 1, 2],
        [1, 2, 1, 1, 3, 3, 1, 4, 1],
        [4, 1, 3, 2, 1, 2, 1, 5, 2],
        [3, 2, 1, 2, 4, 2, 3, 2, 1]
    ],[
        [14, 14, 2, 4, 3, 1, 4, 2, 4],
        [14, 14, 4, 3, 1, 2, 1, 3, 3],
        [3, 4, 1, 5, 2, 4, 1, 2, 5],
        [5, 5, 4, 5, 2, 5, 5, 4, 4],
        [4, 1, 2, 3, 1, 2, 3, 4, 2],
        [4, 1, 4, 4, 2, 4, 1, 3, 4],
        [2, 4, 3, 3, 5, 5, 4, 1, 2],
        [1, 2, 1, 1, 3, 3, 1, 4, 1],
        [4, 1, 3, 2, 1, 2, 1, 5, 2],
        [3, 2, 1, 2, 4, 2, 3, 2, 1]
    ],[
        [14, 14, 2, 4, 3, 1, 4, 2, 4],
        [14, 14, 4, 3, 1, 2, 1, 3, 3],
        [3, 4, 1, 5, 2, 4, 1, 2, 5],
        [5, 5, 4, 5, 2, 5, 5, 4, 4],
        [4, 1, 2, 3, 1, 2, 3, 4, 2],
        [4, 1, 4, 4, 2, 4, 1, 3, 4],
        [2, 4, 3, 3, 5, 5, 4, 1, 2],
        [1, 2, 1, 1, 3, 3, 1, 4, 1],
        [4, 1, 3, 2, 1, 2, 1, 5, 2],
        [3, 2, 1, 2, 4, 2, 3, 2, 1]
    ]))
        model = DQN(len(matrix_3_channels.shape), 161).to(DEVICE)
        print(matrix_3_channels.shape)
        input_tensor = torch.tensor(matrix_3_channels, dtype=torch.float).to(DEVICE)

        output = model(input_tensor)
        
        # find a way to return what the model thinks is the actual prediction of the index of the move that we should make
        max_value, max_index = torch.max(output, dim=0)
        print(float(max_value))
        print(int(max_index))

    def test_model_1d_matrix(self):
        game = Match3Env(90)

        model = DQN(1, 161).to(DEVICE)
        
        input_tensor = torch.tensor(game.return_board, dtype=torch.float).to(DEVICE)
        print(input_tensor.shape)

        output = model(input_tensor)
        
        # find a way to return what the model thinks is the actual prediction of the index of the move that we should make
        max_value, max_index = torch.max(output, dim=0)
        print(float(max_value))
        print(int(max_index))

    def tet_training(self):
        game = Match3Env(90)
        num_actions = 161

        epsilon = 1
        memory = ReplayMemory(1000)

        # make policy and target networks
        policy_dqn = DQN(num_channels=1, out_actions=num_actions)
        target_dqn = DQN(num_channels=1, out_actions=num_actions)

        # make the target and policy network the same
        target_dqn.load_state_dict(policy_dqn.state_dict())


        # define the optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.001)

        for i in range(1000):
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = np.random.randint(0,161)
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(game.return_board).argmax().item()

        print("working")
        
if __name__ == '__main__':
    unittest.main()