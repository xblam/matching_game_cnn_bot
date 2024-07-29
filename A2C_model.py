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


class A2C(nn.Module):
    def __init__(self, in_channels, num_ouputs):
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
            nn.Linear(in_features=128, out_features=num_ouputs)
        )

    def forward(self, x):
        # manually add a layer if inputted matrix is 2 dimensions, or without a batch
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        x = self.conv_block1(x)
        x = x.flatten()
        x = self.layer_stack(x)
        return x
        