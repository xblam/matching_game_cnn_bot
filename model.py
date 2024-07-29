import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
from torch import nn
from gym_match3.envs.match3_env import Match3Env
from display.pygame_display import *
import wandb


DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("USING CUDA")
pin_memory = False
# Define model
class DQN(nn.Module):
    def __init__(self, in_channels, out_actions):
        super().__init__()

        # this model will take in 
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
    learning_rate = 0.01
    discount = 0.9
    network_sync_rate = 25
    memory_size = 100000
    mini_batch_size = 100

    loss_fn = nn.MSELoss() 
    optimizer = None

    def get_state(self, obs):
        # will add the other matrices representing the bomsb and stuff later, but right now just make sure that this shit runs

        # give the color of the gems (1-5), the position of the monster(13)
        # power_ups = torch.stack([obs[6], obs[7],obs[8], obs[9],obs[10]])
        # sum_power_ups = torch.sum(power_ups, dim = 0).unsqueeze(0)

        obs_input_layers = obs[[1,2,3,4,5,6,7,8,9,10,13]]

        # final_state = torch.cat((non_powerups, sum_power_ups), dim = 0)

        # return final_state
        return obs_input_layers

    
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
    
    def train(self, episodes, num_channels, log = False, display = False, render=False, is_slippery=False):
        num_actions = 161
        epsilon = 1

        memory = ReplayMemory(self.memory_size)

        # make policy and target networks
        policy_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        rewards_per_episode = []
        damage_per_episode = []

        step_count=0

        if log:
            wandb.init(
                project="match3"
            )
            
        # each episode represents one life that the system plays
        for i in range(episodes):
            print("NEW LIFE STARTED")
            episode_total_reward = 0
            episode_damage_user = 0

            # in the future when the model is doing better, switch this so that the level changes after every life
            env = Match3Env(90)
            obs, infos = env.reset()
            state = self.get_state(obs)
            pygame.init()
            if display:
                # Initialize the display with the initial state
                matrix = np.array(env.return_game_matrix)
                display = Display(matrix)

            # set the array of valid moves and force the ai to pick from them
            

            # have to see if there is any way to tell if the player's hp is under 0 or the creep's hp is under 0 so then we end the game
            # will probably make it so we end the game if there is more than 50 moves and we still have not won or lost yet

            episode_over = False
            # each step in game
            while not episode_over:

                # choose a move from the list of valid moves (masked output of NN to be the same)
                valid_moves = [index for index, value in enumerate(infos['action_space']) if value == 1]
                if np.random.rand() < epsilon:
                    action = np.random.choice(valid_moves)
                else:
                    with torch.no_grad():
                        input_tensor = state.to(DEVICE)
                        q_values = policy_dqn(input_tensor)
                        valid_mask = torch.zeros_like(q_values)
                        valid_mask[valid_moves] = 1
                        masked_q_values = q_values * valid_mask
                        action = (masked_q_values).argmax().item()

                # take action and get new state
                obs, reward, episode_over, infos = env.step(action)
                print("episode over", episode_over)
                new_state = self.get_state(obs)

                pts_reward = reward['match_damage_on_monster'] + reward['power_damage_on_monster']
                if episode_over:
                    pts_reward += reward["game"]
                    if (reward['game'] > 0):
                        print("THE MONSTER HAS BEEN KILLED")

                episode_total_reward += pts_reward
                episode_damage_user += reward['damage_on_user']
                step_count+=1

                if display:
                    # animate the change
                    (row1,col1), (row2,col2) = self.action_to_coords(action)
                    display.animate_switch((row1,col1),(row2,col2), matrix)

                    # update the matrix and display new state
                    matrix = np.array(env.return_game_matrix)
                    display.update_display(matrix)
                    pygame.time.wait(200)

                # store the replay in memory - probably need to fine tune what we actually want to store
                memory.append((state, action, new_state, pts_reward, episode_over))

                # update the state and step count and reward of the current game
                state = new_state
                print("pts_reward: ", pts_reward)
                print("rewards: ", reward)
                print("steps since last sync: ", step_count)

            rewards_per_episode.append(episode_total_reward)
            damage_per_episode.append(episode_damage_user)
            # Check if enough experience has been collected and if at least 1 reward has been collected
            # change this because we could probably implement it so that we only match to where there is a valid move
            if len(memory)>self.mini_batch_size:
                print("optimizing NN")
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1/(episodes*0.9), 0)

                # sync the policy network with the target network after certain amount of movse
                if step_count > self.network_sync_rate:
                    print('syncing the networks')
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0
            
            if log:
                wandb.log({
                    "running average reward (last 10)": np.sum(rewards_per_episode[-10:])/10,
                    "reward": episode_total_reward,
                    "episodes": i,
                    "epsilon": epsilon,
                    "damange to user": np.sum(damage_per_episode[-10:])/10
                })

        env.close()
        torch.save(policy_dqn.state_dict(), "gym_match3.pt")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # make the policy
        policy_q_list = []
        target_q_list = []
        # state and new state will just be the entire board for now
        for state, action, new_state, pts_reward, episode_over in mini_batch:
            if episode_over:
                target = torch.tensor([pts_reward]).to(DEVICE)
            else:
                with torch.no_grad():
                    input_tensor = new_state.to(DEVICE)
                    target = torch.tensor(
                        pts_reward + self.discount * target_dqn(input_tensor).max()
                    ).to(DEVICE)

            # Get the q value for state from the policy network
            input_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE)
            policy_q = policy_dqn(input_tensor).to(DEVICE)
            policy_q_list.append(policy_q)

            # get the same value for the target network
            
            # target q is the value we output for that specific state
            target_q = target_dqn(input_tensor)
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(policy_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':

    bot = Match3AI()
    # bot.train(10, 11, False, True)

    # run wandb and no display (faster training)
    bot.train(500, 11,True)