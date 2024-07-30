import numpy as np
from collections import deque
import random
import torch
from torch import nn
from gym_match3.envs.match3_env import Match3Env
from display.pygame_display import *
import wandb
import os


counter_file = "run_counter.txt"
def read_counter(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                return int(file.read().strip())
            except ValueError:
                return 0
    else:
        return 0
    
def write_counter(file_path, count):
    with open(file_path, 'w') as file:
        file.write(str(count))
    

DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()): print("USING CUDA")

# Define model
class DQN(nn.Module):
    def __init__(self, in_channels, out_actions):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # the output will be 10 images of dimension 5x4
        )
        self.layer_stack = nn.Sequential(
            # After flattening the matrix into a vector, pass it to the output layer. To determine the input shape, use the print() statement in forward()
            # in this case the input should simply be the size of a singular image
            nn.Linear(in_features=400, out_features=256),
            nn.Linear(in_features=256, out_features=out_actions)
        )

    def forward(self, x):
        # manually add a layer if inputted matrix is 2 dimensions, or without a batch
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        x = self.conv_block1(x)
        x = x.flatten()
        x = self.layer_stack(x)
        return x


class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class Match3AI():
    # hyperparameters
    learning_rate = 0.001
    discount = 0.9
    network_sync_rate = 25
    memory_size = 100000
    mini_batch_size = 100

    loss_fn = nn.MSELoss() 
    optimizer = None

    def get_state(self, obs):
        obs_input_layers = obs[[1,2,3,4,5,6,7,8,9,10,13]]
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
    
    def save_checkpoint(self, save_states, run_id):
        print("saving checkout ---->>>")
        torch.save(save_states, f"{run_id}_parameters.txt")

    def load_checkpoint(self, checkpoint, target, policy, optimizer):
        print("loading checkpoint ---->>>")
        target.load_state_dict(checkpoint['target_state'])
        policy.load_state_dict(checkpoint['policy_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    def train(self, episodes, num_channels, log = False, display = False, render=False, load_model=False, model_id = 0):
        num_actions = 161
        epsilon = 1
        memory = ReplayMemory(self.memory_size)

        # make policy and target networks and optimizer
        policy_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        # this is the counter that will give each of our runs a unique id
        run_id = read_counter(counter_file)
        write_counter(counter_file, run_id+1)

        # NOTE: when you load the model, the saved updated version of model you run will not be saved back to the same file. 
        # This is to allow us to run experiments without the fear of messing up the parameters of the pretrained models.
        if load_model:
            self.load_checkpoint(torch.load(f"{model_id}_parameters.txt"), target_dqn, policy_dqn, self.optimizer)

        if log: wandb.init(project="match3", name = str(run_id))
        
        # each episode represents one life that the system plays
        for i in range(episodes):
            print("NEW LIFE STARTED")
            episode_total_reward = 0
            episode_damage_user = 0
            
            # in the future when the model is doing better, switch this so that the level changes after every life
            env = Match3Env(90)
            obs, infos = env.reset()
            state = self.get_state(obs)
            
            if display:
                pygame.init()
                matrix = np.array(env.return_game_matrix)
                display = Display(matrix)

            step_count = 0

            episode_over = False
            while not episode_over:
                # choose a move from the list of valid moves (masked output of NN to only be valid moves)
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
                action = torch.tensor(action).to(DEVICE)

                obs, reward, episode_over, infos = env.step(action)
                print("episode over", episode_over)
                new_state = self.get_state(obs).to(DEVICE)

                pts_reward = torch.tensor(reward['match_damage_on_monster'] + reward['power_damage_on_monster']).to(DEVICE)
                if episode_over:
                    pts_reward += reward["game"]

                if display:
                    (row1,col1), (row2,col2) = self.action_to_coords(action)
                    display.animate_switch((row1,col1),(row2,col2), matrix)
                    matrix = np.array(env.return_game_matrix)
                    display.update_display(matrix)
                    pygame.time.wait(200)

                memory.append((state, action, new_state, pts_reward, episode_over))
                state = new_state.to(DEVICE)
                step_count+=1
                episode_total_reward += pts_reward
                episode_damage_user += reward['damage_on_user']
                print("pts_reward: ", pts_reward)
                print("rewards: ", reward)
                print("steps since last sync: ", step_count)
                print("RUN ID: ", run_id)

            if len(memory)>self.mini_batch_size:
                print("optimizing NN")
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                epsilon = max(epsilon - 1/(episodes*0.9), 0)
                print('syncing the networks')
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count=0
            
            if log: wandb.log({"reward":episode_total_reward, "episodes":i, "epsilon":epsilon, "damage to user":episode_damage_user})
            
            # XBLAM since the binary files are not that heavy we will just save the policy no matter what
            checkpoint = {'target_state' : target_dqn.state_dict(), 'policy_state' : policy_dqn.state_dict(), 'optimizer' : self.optimizer.state_dict()}
            self.save_checkpoint(checkpoint, run_id)

        env.close()
        torch.save(policy_dqn.state_dict(), "gym_match3.pt")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        policy_q_list = []
        target_q_list = []

        for state, action, new_state, pts_reward, episode_over in mini_batch:
            if episode_over:
                target = torch.tensor([pts_reward]).to(DEVICE)
            else:
                with torch.no_grad():
                    input_tensor = new_state.to(DEVICE)
                    target = torch.tensor(pts_reward + self.discount * target_dqn(input_tensor).max()).to(DEVICE)

            # Get the q value for state from the policy network
            input_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE)
            policy_q = policy_dqn(input_tensor).to(DEVICE)
            policy_q_list.append(policy_q)
            
            # target q is the value we output for that specific state
            target_q = target_dqn(input_tensor)
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(policy_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    bot = Match3AI()
    # train(episodes, num_channels, log = False, display = False, render=False, load_model=False, model_id = 0
    # bot.train(10, 11, False)
    bot.train(episodes=1000, num_channels=11,log=False, display=False, model_id=0)