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
            nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=20*9*10, out_features=256),
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
    
    def train(self, episodes, log=False, load_model=False, model_id=0, retrain = False):
        highest_level = current_level = 0
        num_actions = 161
        num_channels = 11
        epsilon = 1
        if retrain:
            epsilon = 0.5
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

        max_damage = 0

        for i in range(episodes): # each episode of the game
            print("NEW LIFE STARTED")
            damage_taken = 0
            if current_level == 0:
                damage_dealt = 0
                env = Match3Env() # reset the game to the original state
            obs, infos = env.reset()
            state = self.get_state(obs).to(DEVICE)

            episode_over = False
            while not episode_over: # each step of the game
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

                # only reason we need this is to pass into memory
                step_damage_dealt = reward['match_damage_on_monster'] + reward['power_damage_on_monster'] # in some sense this is the only metric that really matters to the network
                damage_dealt += step_damage_dealt
                damage_taken += reward['damage_on_user']

                memory.append((state, action, new_state, step_damage_dealt, episode_over))

                state = new_state
                print("rewards: ", reward)
                print("max damage: ", max_damage)
                print("RUN ID: ", run_id)
        
            if reward['game'] > 0: current_level += 1
            else: current_level = 0
            if current_level > highest_level: highest_level = current_level
            episode_reward = damage_dealt + reward['game']
            if len(memory) > self.mini_batch_size:
                print("optimizing NN")
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                epsilon = max(epsilon - 1 / (episodes * 0.9), 0)
                print('syncing the networks')
                target_dqn.load_state_dict(policy_dqn.state_dict())

            if log: wandb.log({"damage_dealt": damage_dealt, "episodes": i, "epsilon": epsilon, "damage taken": damage_taken, 'highest damage': max_damage, "game_reward": reward['game'], "total_reward_episode" : episode_reward, "highest_level" : highest_level, "current_level":current_level})
            
            if max_damage <= damage_dealt:
                checkpoint = {'target_state': target_dqn.state_dict(), 'policy_state': policy_dqn.state_dict(), 'optimizer': self.optimizer.state_dict()}
                self.save_checkpoint(checkpoint, run_id)
                max_damage = damage_dealt
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
    

    def test(self, episodes, log=False, display=False, model_id=0):
        current_level = highest_level = 0
        num_actions = 161
        num_channels = 11

        # make the target and policy networks
        target_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        policy_dqn = DQN(in_channels=num_channels, out_actions=num_actions).to(DEVICE)
        # set out optmizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        # get the file path and then load out model of choice
        file_path = os.path.join("model_state_dicts", f"{model_id}_state_dict.pth")
        self.load_checkpoint(file_path, target_dqn, policy_dqn, self.optimizer)
        
        run_id = read_counter(counter_file)
        write_counter(counter_file, run_id + 1)

        if log: wandb.init(project="match3_results", name=f"model: {model_id}, run: {str(run_id)}")

        for i in range(episodes):

            num_steps = 0
            if current_level == 0:
                env = Match3Env()
                print("NEW LIFE STARTED")
            obs, infos = env.reset() 
            if display:
                pygame.init()
                matrix = np.array(env.return_game_matrix)
                game_display = Display(matrix)

            state = self.get_state(obs).to(DEVICE)
            # for the decision making process we just keep it the same
            episode_over = False
            while not episode_over:
                valid_moves = [index for index, value in enumerate(infos['action_space']) if value == 1]
                with torch.no_grad():
                    input_tensor = state.unsqueeze(0)
                    q_values = policy_dqn(input_tensor)
                    valid_q_values = q_values[0, valid_moves]
                    action = valid_moves[valid_q_values.argmax().item()]
                action = torch.tensor(action).to(DEVICE)

                # still need this because we need obs to get the next state
                obs, reward, episode_over, infos = env.step(action)
                state = self.get_state(obs).to(DEVICE)
                num_steps += 1
                print("step number:", num_steps)
                print("current_level:", current_level)
                print("highest_level:", highest_level)
                print(reward)
                print("run_id:", run_id)
                if display:
                    (row1,col1), (row2,col2) = self.action_to_coords(action)
                    game_display.animate_switch((row1,col1),(row2,col2), matrix)
                    matrix = np.array(env.return_game_matrix)
                    game_display.update_display(matrix)
                    pygame.time.wait(100)
            if reward['game'] > 0: current_level += 1
            else: current_level = 0
            if current_level > highest_level: highest_level = current_level

            if log: wandb.log({"current_level:":current_level, "highest_level:":highest_level, "episode:": i}) 
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", "--run_test", action='store_true')
    parser.add_argument("-train", "--run_train", action="store_true")
    parser.add_argument("-retrain", "--run_retrain", action="store_true")
    parser.add_argument("-e", "--episodes", type=int, required=True)
    parser.add_argument("-l", "--log", action='store_true')
    parser.add_argument("-d", "--display", action='store_true')
    parser.add_argument("-ldmd", "--load_model", type=int, help="put the id of the pretrained model you want to load") 
    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    print('testing:', args.run_test)
    print('training:', args.run_train)
    print('retraining:', args.run_retrain)
    print('episodes:', args.episodes)
    print('log:', args.log)
    print('display:', args.display)
    print('loading_model:', args.load_model)

    bot = Match3AI()
    if (args.run_test): 
        # run this with -t, -e, and -ldm
        print("RUNNING TEST FUNCTION")
        bot.test(args.episodes, args.log, args.display, args.load_model)
    else: 
        # only need to run this with -e
        print("RUNNING RETRAIN") if args.run_retrain else print("RUNNING TRAIN")
        load_model = True if (args.load_model) else False
        model_id = args.load_model
        bot.train(args.episodes, args.log, load_model, model_id, args.run_retrain)

            
    # if ()
    # bot.train(args.episodes, args.log, args.load_model, args.model_id)

if __name__ == '__main__':
    main()
    # bot = Match3AI()
    # # train(episodes, log = False, display = False, render=False, load_model=False, model_id = 0
    # # bot.train(10, 11, False)
    # bot.train(episodes=1000, log=True, display=False, load_model=False, model_id=28)

