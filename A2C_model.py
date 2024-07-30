import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define Actor Network
class ActorNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# Define Critic Network
class CriticNN(nn.Module):
    def __init__(self, input_dim):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(num_episodes = int):
    actor = ActorNN()
    critic = CriticNN()


    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action_probs = ActorNN(state)
            value = CriticNN(state)      
            action = sample_action(action_probs)
            next_state, reward, done, _ = env.step(action)      
            next_value = CriticNN(next_state)      
            advantage = reward + (1 - done) * gamma * next_value - value      
            actor_loss = -log_prob(action) * advantage + entropy_coeff * entropy(action_probs)
            critic_loss = advantage^2      
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()      
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()      
            state = next_state

if __name__ == "__main__":
    train_model(1000)