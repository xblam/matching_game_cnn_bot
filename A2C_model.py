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
    def __init__(self, in_channels, num_ouputs)