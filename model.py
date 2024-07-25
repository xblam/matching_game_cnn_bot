import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
import time
from torch import nn
from gym_match3.envs.match3_env import Match3Env
from display.pygame_display import *


DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
class DQN(nn.Module):
    def __init__(self, in_channels, out_actions):
        super().__init__()

        # this model is designed to take in inputs of rbg images with dimensinos 10x9
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # the output will be 10 images of dimension 5x4
        )

        self.layer_stack = nn.Sequential(
            # After flattening the matrix into a vector, pass it to the output layer. To determine the input shape, use the print() statement in forward()

            # in this case the input should simply be the size of a singular image
            nn.Linear(in_features=200, out_features=128),
            nn.Linear(in_features=128, out_features=out_actions)
        )

        # if something happens and we dont input the batch_size, then we will just manually resize the array so that the model still runs

    def forward(self, x):
        # manually add a layer if inputted matrix is 2 dimensions, or without a batch
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        x = self.conv_block1(x)
        x = x.flatten()
        x = self.layer_stack(x)
        return x


# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


# FrozeLake Deep Q-Learning
class Match3AI():

    # Hyperparameters (adjustable)
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 25
    replay_memory_size = 10000
    mini_batch_size = 100

    loss_fn = nn.MSELoss() 
    optimizer = None

    def get_state(self, obs):
        # will add the other matrices representing the bomsb and stuff later, but right now just make sure that this shit runs
        return obs[[1,2,3,4,5,13,24]]
    
    def action_to_coords(self, action):
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
    
    def train(self, episodes, num_channels, render=False, is_slippery=False):
        num_actions = 161
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        # make policy and target networks
        policy_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)

        epsilon_history = []

        step_count=0

        # each episode represents one life that the system plays

        for i in range(episodes):
            print("NEW LIFE STARTED")
            env = Match3Env(90)
            obs, infos = env.reset()
            state = self.get_state(obs)
            pygame.init()

            # Initialize the display with the initial state
            matrix = np.array(env.return_game_matrix)
            display = Display(matrix)
            damage = 0


            terminated = False # agent took too many steps

            # have to see if there is any way to tell if the player's hp is under 0 or the creep's hp is under 0 so then we end the game
            # will probably make it so we end the game if there is more than 50 moves and we still have not won or lost yet

            # while(not terminated): # for each step in the episode
            while not terminated:


                if random.random() < epsilon:
                    action = np.random.randint(0,161)
                else:
                    with torch.no_grad(): # not really sure why with torch.no_grad is needed
                        # make sure you put the entirity of all of the layers that you want to pass here
                        input_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE)
                        action = policy_dqn(input_tensor).argmax().item()

                # animate the change
                (row1,col1), (row2,col2) = self.action_to_coords(action)
                display.animate_switch((row1,col1),(row2,col2), matrix)

                # take the action and observe the next state and reward
                # # XBLAM probably have to see what the game environment returns to set up these figures
                obs, reward, dones, infos = env.step(action)

                pts_reward = reward['score'] + reward['match_damage_on_monster']*10 + reward['power_damage_on_monster']*10
                damage += reward['damage_on_user']
                step_count+=1



                if damage > 20:
                    terminated = True

                # obs has been updated so now we can get the new state
                new_state = self.get_state(obs)
                matrix = np.array(env.return_game_matrix)
                display.update_display(matrix)
                pygame.time.wait(100)

                # store the replay in memory - probably need to fine tune what we actually want to store
                memory.append((state, action, new_state, pts_reward, terminated))
                # update the state and step count and reward of the period

                rewards_per_episode[i] += pts_reward
                print("damage: ", damage)
                print("pts_reward: ", pts_reward)
                print(rewards_per_episode[i])
                print("rewards: ", reward)

            # Check if enough experience has been collected and if at least 1 reward has been collected
            # change this because we could probably implement it so that we only match to where there is a valid move
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                print("optimizing NN")
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # sync the policy network with the target network after certain amount of movse
                if step_count > self.network_sync_rate:
                    print('syncing the networks')
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0
            
            print("total reward of episode: " , rewards_per_episode)

        env.close()
        torch.save(policy_dqn.state_dict(), "gym_match3.pt")

        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []
        # state and new state will just be the entire board for now
        for state, action, new_state, pts_reward, terminated in mini_batch:

            if terminated:
                # if this runs, then we just make the target the reward we just got
                target = torch.tensor([pts_reward]).to(DEVICE)
            else:
                # otherwise choose the best action
                with torch.no_grad():
                    input_tensor = torch.tensor(new_state, dtype=torch.float).to(DEVICE)
                    target = torch.tensor(
                        # make the new state into a tensor input
                        pts_reward + self.discount_factor_g * target_dqn(input_tensor).max()
                    ).to(DEVICE)

            # Get the current set of Q values
            input_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE)
            current_q = policy_dqn(input_tensor)
            current_q_list.append(current_q)

            # Get the target set of Q values
            input_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE)
            target_q = target_dqn(input_tensor)
            
            # Adjust the specific action to the target that was just calculated. 
            # Target_q[batch][action], hardcode batch to 0 because there is only 1 batch.
            target_q[0][action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_tensor(self, state):
        return(torch.tensor(state, dtype=torch.float).to(DEVICE))
    
    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_channels=3, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql_cnn.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):

        # Loop each state and print policy to console
        for s in range(16):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s))[0].tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input()).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':

    bot = Match3AI()
    bot.train(100,7)
    
#     model = DQN(1, 161).to(DEVICE)
#     matrix = np.array(([
#     [14, 14, 2, 4, 3, 1, 4, 2, 4],
#     [14, 14, 4, 3, 1, 2, 1, 3, 3],
#     [3, 4, 1, 5, 2, 4, 1, 2, 5],
#     [5, 5, 4, 5, 2, 5, 5, 4, 4],
#     [4, 1, 2, 3, 1, 2, 3, 4, 2],
#     [4, 1, 4, 4, 2, 4, 1, 3, 4],
#     [2, 4, 3, 3, 5, 5, 4, 1, 2],
#     [1, 2, 1, 1, 3, 3, 1, 4, 1],
#     [4, 1, 3, 2, 1, 2, 1, 5, 2],
#     [3, 2, 1, 2, 4, 2, 3, 2, 1]
# ]))
#     print(matrix.shape)
    
#     input_tensor = torch.tensor(matrix, dtype=torch.float).to(DEVICE)

#     output = model(input_tensor)


#     # Print resulting predictions
#     print(output)
    
#     # find a way to return what the model thinks is the actual prediction of the index of the move that we should make
#     max_value, max_index = torch.max(output, dim=0)

#     print(int(max_index))

    