import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gym_match3.envs.match3_env import Match3Env
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import wandb

DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")
if (T.cuda.is_available()):
    print("RUNNING WITH CUDA")

counter_file = "a2c_run_counter.txt"
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
    
# Define Actor Network
class ActorNN(nn.Module):
    def __init__(self):
        super(ActorNN, self).__init__()
        self.conv1 = nn.Conv2d(11,32,kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*9*10, 256)
        self.fc2 = nn.Linear(256, 161) # set it since we know how many outputs we want already
    
    def forward(self, x):
        # no pooling needed
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution

# Define Critic Network
class CriticNN(nn.Module): # keep in mind that this is a q-value based critic
    def __init__(self):
        super(CriticNN, self).__init__()
        self.conv1 = nn.Conv2d(11,32,kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*9*10, 256) # just make sure that the dimensions are correct
        self.fc2 = nn.Linear(256, 1) # just set this to 1 since it outputs the value function
    
    def forward(self, x):
        # no pooling needed
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   

# class to initiate and train the agent
class A2CModel():     
    def __init__(self):
        self.learning_rate = 0.001
        self.gamma = 0.99

    def get_state(self, obs):
        obs_input_layers = obs[[1,2,3,4,5,6,7,8,9,10,13]]
        return obs_input_layers
    
    def show_histogram(self, distribution):
        flat_tensor = distribution.detach().numpy().flatten()
        plt.hist(flat_tensor, bins = 40, alpha=0.7, color='blue', edgecolor='black')
        plt.show()

    def compute_returns(self, step_return, rewards, masks, gamma=.99):
        returns = []
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            step_return = reward + gamma*step_return*mask 
            returns.append(step_return)
        returns.reverse()
        return returns
    
    def save_checkpoint(self, save_states, run_id):
        print("saving checkpoint ---->>>")
        dir_name = "a2c_state_dicts"
        os.makedirs(dir_name, exist_ok=True)
        file_path = os.path.join(dir_name, f"{run_id}_state_dict.pth")
        T.save(save_states, file_path)

    def load_checkpoint(self, save_file_path, actor, critic, actor_optimizer, critic_optimizer):
        print("loading checkpoint ---->>>")
        state_dict = T.load(save_file_path)
        actor.load_state_dict(state_dict['actor_state'])
        critic.load_state_dict(state_dict['critic_state'])
        actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
    
    def train(self, num_episodes = 1, log = False, load_model = False, model_id=0):
        highest_level = current_level = 0 # initializing the variables

        run_id = read_counter(counter_file) # for all files we will be assigning an id to the run
        write_counter(counter_file, run_id + 1)

        if log: wandb.init(project="match3_a2c", name=str(run_id))

        max_damage = 0
        self.actor = ActorNN().to(DEVICE)
        self.critic = CriticNN().to(DEVICE)
        self.learning_rate = 0.001
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr = self.learning_rate)

        for current_episode in range(num_episodes): # each episode will be one playthrough of a levela
            print('STARTED NEW LIFE')
            log_prob_list = []
            value_list = []
            reward_list = []
            mask_list = []
            episode_damage = 0

            if (current_level == 0):
                env = Match3Env() # we want to increment the level 
            obs,infos = env.reset() #XBLAM can change this to underscore later if need be

            episode_over = False
            while not episode_over: # iterate over life 

                state = self.get_state(obs).unsqueeze(0)
                state = state.to(DEVICE)

                distribution, value = self.actor(state), self.critic(state)
                
                distribution_probs = distribution.probs # get the actual probabilities
                valid_moves = T.tensor(infos['action_space']).to(DEVICE)
                masked_distribution = distribution_probs*valid_moves
                new_distribution = T.distributions.Categorical(probs=masked_distribution)
                # action = new_distribution.sample() # get the action 
                action = distribution.sample()

                # play the game and update the state
                obs, reward, episode_over, infos = env.step(action)
                new_state = self.get_state(obs).unsqueeze(0).to(DEVICE)
                
                log_prob = distribution.log_prob(action).unsqueeze(0)
                log_prob_list.append(log_prob)
                value_list.append(value)
                
                step_damage = reward['power_damage_on_monster'] + reward['match_damage_on_monster'] 
                episode_damage += step_damage
                
                reward_list.append(T.tensor([step_damage]).to(DEVICE))
                mask_list.append(T.tensor([1-episode_over]).to(DEVICE))
                
                print("step damage:", step_damage)
                print()
                print("current episode:", current_episode)
                state = new_state
            
            # CODE UNDER RUNS WHEN THE EPISODE IS OVER
            if log: wandb.log({"episode_damage":episode_damage, "current_level":current_level, "episode":current_episode})
            
            if max_damage <= episode_damage: # save parameters of the best model
                checkpoint = {'actor_state': self.actor.state_dict(), 'critic_state': self.critic.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict()}
                self.save_checkpoint(checkpoint, run_id)
                max_damage = episode_damage
                print("SAVED PARAMETERS TO FOLDER")

            if reward['game'] > 0:
                print('MONSTER DIED')
                current_level += 1
            else: current_level = 0

            # optimizing the model
            next_value = self.critic(new_state)
            returns = self.compute_returns(next_value, reward_list, mask_list)

            log_prob_list = T.cat(log_prob_list)
            value_list = T.cat(value_list)
            returns = T.cat(returns).detach() # not really sure why we have to detach it

            # TODO figure out what all this mess means, and if there is any way in which you can optimize this
            advantage = returns - value_list
            actor_loss = -(log_prob_list*advantage.detach()).mean()
            print('ACTOR LOSS:', actor_loss)
            critic_loss = advantage.pow(2).mean()
            print('CRITIC LOSS:', critic_loss)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int)
    parser.add_argument('-l', '--log', action="store_true")
    parser.add_argument('-ldmd', '--load_model', action='store_true')

    args = parser.parse_args()

    print("episodes:", args.episodes)
    print("log:", args.log)

    bot = A2CModel()
    bot.train(args.episodes, args.log)

if __name__ == "__main__":
    main()


    


# the actor first does an action (do not need epislon
# update the observation of the state and the reward
# temporal difference learning?
# give the observation to critic
# get val function
# once we have the val function we can get the advantage