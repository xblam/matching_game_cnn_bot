import numpy as np
from collections import deque
import random
import torch
from torch import nn
from gym_match3.envs.match3_env import Match3Env
from display.pygame_display import *
import wandb
import os
import argparse


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
    

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): print("USING CUDA")

# Define model
class DQN(nn.Module):
    def __init__(self, in_channels, out_actions):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=200, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_actions)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        x = self.conv_block1(x)
        x = x.flatten(start_dim=1)
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
    network_sync_rate = 50
    memory_size = 100000
    mini_batch_size = 100

    loss_fn = nn.MSELoss() 
    optimizer = None

    def get_state(self, obs):
        obs_input_layers = obs[[1,2,3,4,5,6,7,8,9,10,13]]
        return obs_input_layers

    def action_to_coords(self, action):
        if action < 80:
            row = action // 8
            col = action % 8
            coord1 = (row, col)
            coord2 = (row, col + 1)
        else:
            action = action - 80
            row = action // 9
            col = action % 9
            coord1 = (row, col)
            coord2 = (row + 1, col)
        return coord1, coord2
    
    def save_checkpoint(self, save_states, run_id):
        print("saving checkpoint ---->>>")
        dir_name = "model_state_dicts"
        os.makedirs(dir_name, exist_ok=True)
        file_path = os.path.join(dir_name, f"{run_id}_state_dict.pth")
        torch.save(save_states, file_path)

    def load_checkpoint(self, save_file_path, target, policy, optimizer):
        print("loading checkpoint ---->>>")
        state_dict = torch.load(save_file_path)
        target.load_state_dict(state_dict['target_state'])
        policy.load_state_dict(state_dict['policy_state'])
        optimizer.load_state_dict(state_dict['optimizer'])
    
    def train(self, episodes, log=False, display=False, load_model=False, model_id=0):
        num_actions = 161
        num_channels = 11
        epsilon = 1
        memory = ReplayMemory(self.memory_size)

        policy_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        run_id = read_counter(counter_file)
        write_counter(counter_file, run_id + 1)

        if load_model:
            file_path = os.path.join("model_state_dicts", f"{model_id}_state_dict.pth")
            self.load_checkpoint(file_path, target_dqn, policy_dqn, self.optimizer)

        if log: wandb.init(project="match3", name=str(run_id))

        max_reward = 0

        for i in range(episodes):
            print("NEW LIFE STARTED")
            episode_total_reward = 0
            episode_damage_user = 0
            
            env = Match3Env()
            obs, infos = env.reset()
            state = self.get_state(obs).to(DEVICE)
            step_count = 0
            
            if display:
                pygame.init()
                matrix = np.array(env.return_game_matrix)
                display = Display(matrix)

            episode_over = False

            game_won = 0
            while not episode_over:
                valid_moves = [index for index, value in enumerate(infos['action_space']) if value == 1]
                if np.random.rand() < epsilon:
                    action = np.random.choice(valid_moves)
                else:
                    with torch.no_grad():
                        input_tensor = state.unsqueeze(0).to(DEVICE)
                        q_values = policy_dqn(input_tensor)
                        valid_q_values = q_values[0, valid_moves]
                        action = valid_moves[valid_q_values.argmax().item()]
                
                action = torch.tensor(action).to(DEVICE)

                obs, reward, episode_over, infos = env.step(action)
                new_state = self.get_state(obs).to(DEVICE)

                pts_reward = reward['match_damage_on_monster'] * 5 + reward['power_damage_on_monster'] * 5
                if episode_over:
                    pts_reward += reward["game"]
                    if reward['game'] > 0:
                        game_won = 1

                if display:
                    (row1, col1), (row2, col2) = self.action_to_coords(action)
                    display.animate_switch((row1, col1), (row2, col2), matrix)
                    matrix = np.array(env.return_game_matrix)
                    display.update_display(matrix)
                    pygame.time.wait(200)

                memory.append((state, action, new_state, pts_reward, episode_over))
                state = new_state
                step_count += 1
                episode_total_reward += pts_reward
                episode_damage_user += reward['damage_on_user']
                print("steps since last sync: ", step_count)
                print("pts_reward: ", pts_reward)
                print("rewards: ", reward)
                print("highest reward: ", max_reward)
                print("RUN ID: ", run_id)

            print("optimizing NN")
            mini_batch = memory.sample(self.mini_batch_size)
            self.optimize(mini_batch, policy_dqn, target_dqn)
            epsilon = max(epsilon - 1 / (episodes * 0.9), 0)
            print('syncing the networks')
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0

            print("TOTAL REWARD OF EPISODE: ", episode_total_reward)

            if log: wandb.log({"reward": episode_total_reward, "episodes": i, "epsilon": epsilon, "damage to user": episode_damage_user, 'highest reward': max_reward, "game_won" : game_won})
            
            if max_reward <= episode_total_reward:
                checkpoint = {'target_state': target_dqn.state_dict(), 'policy_state': policy_dqn.state_dict(), 'optimizer': self.optimizer.state_dict()}
                self.save_checkpoint(checkpoint, run_id)
                max_reward = episode_total_reward
                print("SAVED PARAMETERS TO FOLDER")

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, episode_overs = zip(*mini_batch)

        states = torch.stack(states).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        new_states = torch.stack(new_states).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        episode_overs = torch.tensor(episode_overs, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            target_q_values = target_dqn(new_states).max(1)[0]
            target_q_values = rewards + (1 - episode_overs) * self.discount * target_q_values

        policy_q_values = policy_dqn(states).gather(1, actions).squeeze()

        loss = self.loss_fn(policy_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def test(self, episodes, num_channels, display=False, model_id=0):
        num_actions = 161
        target_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        policy_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        file_path = os.path.join("model_state_dicts", f"{model_id}_state_dict.pth")
        self.load_checkpoint(file_path, target_dqn, policy_dqn, self.optimizer)

        run_id = read_counter(counter_file)
        write_counter(counter_file, run_id + 1)

        for i in range(episodes):
            print("NEW LIFE STARTED")
            episode_total_reward = 0
            episode_damage_user = 0
            
            env = Match3Env()
            obs, infos = env.reset()
            state = self.get_state(obs)
            step_count = 0
            
            if display:
                pygame.init()
                matrix = np.array(env.return_game_matrix)
                display = Display(matrix)

            episode_over = False
            while not episode_over:
                valid_moves = [index for index, value in enumerate(infos['action_space']) if value == 1]
                with torch.no_grad():
                    input_tensor = state.unsqueeze(0)
                    q_values = policy_dqn(input_tensor)
                    valid_q_values = q_values[0, valid_moves]
                    action = valid_moves[valid_q_values.argmax().item()]
                action = torch.tensor(action)

                obs, reward, episode_over, infos = env.step(action)
                state = self.get_state(obs)

                pts_reward = reward['match_damage_on_monster'] * 5 + reward['power_damage_on_monster'] * 5
                if episode_over:
                    pts_reward += reward["game"]

                if display:
                    (row1, col1), (row2, col2) = self.action_to_coords(action)
                    display.animate_switch((row1, col1), (row2, col2), matrix)
                    matrix = np.array(env.return_game_matrix)
                    display.update_display(matrix)
                    pygame.time.wait(200)

                
                step_count += 1
                episode_total_reward += pts_reward
                episode_damage_user += reward['damage_on_user']
                print("steps since last sync: ", step_count)
                print("pts_reward: ", pts_reward)
                # XBLAM should probably print other things like how much hp the monster has or something
                print("rewards: ", reward)
                print("RUN ID: ", run_id)

        env.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--episodes", type=int, required=True)
    parser.add_argument("-l", "--log", action='store_true')
    parser.add_argument("-d", "--display", action='store_true')  
    parser.add_argument("-ld", "--load_model", action='store_true') 
    parser.add_argument("-mid", "--model_id", type=int, required=True)
    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    print('episodes:', args.episodes)
    print('log:', args.log)
    print('display:', args.display)
    print('load model:', args.load_model)
    print('model id:', args.model_id)
    #  episodes, log=False, display=False, load_model=False, model_id=0:

    bot = Match3AI()

    bot.train(args.episodes, args.log, args.display, args.load_model, args.model_id)

if __name__ == '__main__':
    main()
    # bot = Match3AI()
    # # train(episodes, log = False, display = False, render=False, load_model=False, model_id = 0
    # # bot.train(10, 11, False)
    # bot.train(episodes=1000, log=True, display=False, load_model=False, model_id=28)

