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
import cProfile, pstats

DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")
if (T.cuda.is_available()):
    print("RUNNING WITH CUDA")

counter_file = "a2c_state_dicts/a2c_run_counter.txt"
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
        self.conv1 = nn.Conv2d(26,64,kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*9*10, 512)
        self.fc2 = nn.Linear(512, 161) # set it since we know how many outputs we want already
    
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
        self.conv1 = nn.Conv2d(26,64,kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*9*10, 512) # just make sure that the dimensions are correct
        self.fc2 = nn.Linear(512, 1) # just set this to 1 since it outputs the value function
    
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
        self.gamma = 0.99
        self.actor = ActorNN().to(DEVICE)
        self.critic = CriticNN().to(DEVICE)
        self.learning_rate = 0.001
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr = self.learning_rate)

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
            # mask is essentially whether or not the episode iss over
            step_return = reward + gamma*step_return*mask 
            returns.append(step_return)
        returns.reverse()
        return returns
        '''returns list is a list containing how valuable this reward is, if we also consider all future rewards
        # ex: if a rewards list is [10,20,30], the function will return [523,47,30]'''
    
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
        # in some sense we want the model to learn on many different games at once, so that it diversifies
        current_level = 0 # initializing the variables

        max_damage = 0

        self.log_prob_list, self.value_list, self.reward_list, self.mask_list = [],[],[],[]
        run_id = read_counter(counter_file) # for all files we will be assigning an id to the run
        write_counter(counter_file, run_id + 1)

        run_name = run_id

        if load_model:
            file_path = os.path.join("a2c_state_dicts", f"{model_id}_state_dict.pth")
            self.load_checkpoint(file_path, self.actor, self.critic, self.actor_optimizer, self.critic_optimizer)        
            run_name = f"{run_id}-{model_id}"


        if log: wandb.init(project="match3_a2c", name=str(run_name))
        env = Match3Env() # made the object here

        for current_episode in range(num_episodes): # each episode will be one playthrough of a levela
            print('STARTED NEW LIFE')
            episode_damage = 0
            step_count = 0

            obs,infos = env.randomize_level() # we actually want it to go through every level so just move on when you die 

            episode_over = False
            while not episode_over: # iterate over life 

                state = obs.unsqueeze(0).to(DEVICE)
                distribution, self.value = self.actor(state), self.critic(state)
                valid_moves = T.tensor(infos['action_space']).to(DEVICE)
                masked_distribution = distribution.probs*valid_moves
                new_distribution = T.distributions.Categorical(probs=masked_distribution)
                action = new_distribution.sample() # get the action 

                # play the game and update the state
                obs, reward, episode_over, infos = env.step(action)
                new_state = obs.unsqueeze(0).to(DEVICE)
                
                self.log_prob = distribution.log_prob(action).unsqueeze(0)
                
                step_damage = reward['power_damage_on_monster'] + reward['match_damage_on_monster'] 
                episode_damage += step_damage

                self.reward = T.tensor([step_damage]).to(DEVICE)
                self.mask = T.tensor([1-episode_over]).to(DEVICE)
                
                state = new_state
                step_count += 1

                print("step damage:", step_damage)
                print('episode damage', episode_damage)
                print('max damage:', max_damage)
                print('reward:', reward)
                print("current episode:", current_episode)
                print('step_count:', step_count)
                print('run id:', run_id)
                
                # if the round is over, we will reward the ai with the result
                if episode_over and reward['game'] > 0: self.reward += reward['game'] 
                self.update_model(new_state, self.reward, self.mask, self.log_prob, self.value) # changed this to update after every move so that it learns faster

            # CODE UNDER RUNS WHEN THE EPISODE IS OVER

            if log:wandb.log({'actor loss':self.actor_loss, 'critic loss':self.critic_loss})

            if reward['game'] > 0:
                print('MONSTER DIED')
                current_level += 1
            else: current_level = 0
            
            if max_damage <= episode_damage: # save parameters of the best model
                checkpoint = {'actor_state': self.actor.state_dict(), 'critic_state': self.critic.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict()}
                self.save_checkpoint(checkpoint, run_id)
                max_damage = episode_damage
                print("SAVED PARAMETERS TO FOLDER")

            if log: wandb.log({"episode_damage":episode_damage, "current_level":current_level, "episode":current_episode, 'game reward':reward['game'], 'total reward':reward['game']+episode_damage})

        env.close()

    def update_model(self, new_state, reward, mask, log_prob, value, gamma=0.99):

        # Compute the value for the new state
        future_value = self.critic(new_state)
    
        # Compute returns
        returns = reward + gamma * future_value * mask
        returns = returns.detach()  # Detach to prevent gradient updates through returns

        # Calculate advantage
        advantage = returns - value

        # Calculate losses
        self.actor_loss = -(log_prob * advantage.detach())
        self.critic_loss = advantage.pow(2)
    
        # Print losses for debugging
        print('ACTOR LOSS:', self.actor_loss.item())
        print('CRITIC LOSS:', self.critic_loss.item())
    
        # Zero gradients, perform backpropagation, and update weights
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor_loss.backward()
        self.critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def test(self,num_episodes, log=False, model_id=0):
        print("TESTING")
        file_path = os.path.join("a2c_state_dicts", f"{model_id}_state_dict.pth")
        self.load_checkpoint(file_path, self.actor, self.critic, self.actor_optimizer, self.critic_optimizer)
        current_level, highest_level = 0,0

        if log: wandb.init(project="match3_a2c", name=str(f'TEST: {model_id}'))

        for episode in range(num_episodes):
            if (current_level == 0):
                env = Match3Env() 
            obs,infos = env.reset() 

            episode_damage, step_count = 0, 0
            episode_over = False
            while not(episode_over):
                state = obs.unsqueeze(0).to(DEVICE)
                # just like in the train we will get the move
                distribution = self.actor(state)
                valid_moves = T.tensor(infos['action_space']).to(DEVICE)
                masked_distribution = distribution.probs*valid_moves
                new_distribution = T.distributions.Categorical(probs=masked_distribution)
                action = new_distribution.sample()

                # take the step and record outputs and values
                obs, reward, episode_over, infos = env.step(action)
                step_damage = reward['power_damage_on_monster'] + reward['match_damage_on_monster'] 
                episode_damage += step_damage
                step_count += 1
                print('step damage:', step_damage)
                print('step counter:', step_count)
                print('test_model_id:', model_id)
                print('episode:', episode)
            total_reward = episode_damage + reward['game']
            if log: wandb.log({'episode damage':episode_damage, 'episode': episode, 'current level': current_level, 'game reward':reward['game'], 'total reward':total_reward})
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int)
    parser.add_argument('-l', '--log', action="store_true")
    parser.add_argument('-ldmd', '--load_model', type=int)
    parser.add_argument('-test', '--testing', action='store_true')

    args = parser.parse_args()

    print("episodes:", args.episodes)
    print("log:", args.log)
    print('ldmd:', args.load_model)
    print('testing:', args.testing)

    bot = A2CModel()
    # num_episodes = 1, log = False, load_model = False, model_id=0
    if args.testing:
        bot.test(args.episodes, args.log, args.load_model)
    else: bot.train(args.episodes, args.log, True, args.load_model) if args.load_model else bot.train(args.episodes, args.log, False)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profile_results.prof')


    


# the actor first does an action (do not need epislon
# update the observation of the state and the reward
# temporal difference learning?
# give the observation to critic
# get val function
# once we have the val function we can get the advantage