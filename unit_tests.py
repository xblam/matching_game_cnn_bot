
import unittest
import numpy as np
from model import *
from gym_match3.envs.match3_env import Match3Env
from display.pygame_display import *

class TestModel(unittest.TestCase):
    def state_to_tensor(self, state):
        return(torch.tensor(state, dtype=torch.float).to(DEVICE))

    def test_model_1d_matrix(self):
        game = Match3Env(90)

        model = DQN(1, 161).to(DEVICE)
        
        input_tensor = self.state_to_tensor(game.return_state)
        print(input_tensor.shape)

        output = model(input_tensor)
        
        # find a way to return what the model thinks is the actual prediction of the index of the move that we should make
        max_value, max_index = torch.max(output, dim=0)
        self.assertLess(max_index, 162)
        self.assertLess(max_value, 1)
#        print(output)
#        print(float(max_value))
#        print(int(max_index))


    def test_training(self):

        game = Match3Env(90)
        _last_obs, infos = game.reset()       
        state = game.return_state
        num_actions = 161

        epsilon = 0
        memory = ReplayMemory(1000)

        # make policy and target networks
        policy_dqn = DQN(num_channels=1, out_actions=num_actions).to(DEVICE)
        target_dqn = DQN(num_channels=1, out_actions=num_actions).to(DEVICE)

        # make the target and policy network the same
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # define the optimizer
        for i in range(1):
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            # while(not terminated and not truncated):
            for b in range(10):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = np.random.randint(0,161)
                else:
                    # select best action
                    with torch.no_grad():
                        # although this means that the action will always be the same for a given state, this would only happen if epsilon were equal to 0
                        input_tensor = self.state_to_tensor(game.return_state)
                        action = policy_dqn(input_tensor).argmax().item()
        


        
if __name__ == '__main__':
    unittest.main()