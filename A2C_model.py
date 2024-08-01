import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym_match3.envs.match3_env import Match3Env
import argparse

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
        x = F.softmax(self.fc2(x), dim=-1)
        return x

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

    def train(self, num_episodes = int):
        actor = ActorNN()
        critic = CriticNN()

        current_level = 0
        max_level = 0

        for current_episode in range(num_episodes): # each episode will be one playthrough of a level
            if (current_level == 0):
                env = Match3Env()
            obs,infos = env.reset()

            game_over = False
            while not game_over: # this will indicate each step we take in the game 
                state = self.get_state(obs)
                input_tensor = state.unsqueeze(0)

                action = T.tensor(actor(input_tensor)).argmax().item() # actor takes in state and outputs action
                second_action = T.tensor(critic(input_tensor))

                print("action:", action)
                print('critic:', second_action)


                # critic takes in action and state and outputs value
                # value = critic(action, state)
                print(current_episode)
                obs, reward, episode_over, infos = env.step(action) # we use the action to get the new_state and such
                new_state = self.get_state(obs) # get next state
                episode_reward = reward['power_damage_on_monster'] + reward['match_damage_on_monster']
                print(episode_reward)

                # TODO: have the program do something with the move we selected
                # find a way to get the new state

                game_over = True #XBLAM placeholder for testing, will remove when we have the code up and running.



            # code under this point means that the match has ended

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, required=True)

    args = parser.parse_args()

    print("episodes:", args.episodes)

    bot = A2CModel()
    bot.train(args.episodes)

if __name__ == "__main__":
    main()


    


# the actor first does an action (do not need epislon
# update the observation of the state and the reward
# temporal difference learning?
# give the observation to critic
# get val function
# once we have the val function we can get the advantage