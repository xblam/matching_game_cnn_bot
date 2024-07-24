
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model import *

class TestBoard(unittest.TestCase):
    def test_model_3_layer_matrix(self):
       
       
        matrix_3_channels = np.array(([
        [14, 14, 2, 4, 3, 1, 4, 2, 4],
        [14, 14, 4, 3, 1, 2, 1, 3, 3],
        [3, 4, 1, 5, 2, 4, 1, 2, 5],
        [5, 5, 4, 5, 2, 5, 5, 4, 4],
        [4, 1, 2, 3, 1, 2, 3, 4, 2],
        [4, 1, 4, 4, 2, 4, 1, 3, 4],
        [2, 4, 3, 3, 5, 5, 4, 1, 2],
        [1, 2, 1, 1, 3, 3, 1, 4, 1],
        [4, 1, 3, 2, 1, 2, 1, 5, 2],
        [3, 2, 1, 2, 4, 2, 3, 2, 1]
    ],[
        [14, 14, 2, 4, 3, 1, 4, 2, 4],
        [14, 14, 4, 3, 1, 2, 1, 3, 3],
        [3, 4, 1, 5, 2, 4, 1, 2, 5],
        [5, 5, 4, 5, 2, 5, 5, 4, 4],
        [4, 1, 2, 3, 1, 2, 3, 4, 2],
        [4, 1, 4, 4, 2, 4, 1, 3, 4],
        [2, 4, 3, 3, 5, 5, 4, 1, 2],
        [1, 2, 1, 1, 3, 3, 1, 4, 1],
        [4, 1, 3, 2, 1, 2, 1, 5, 2],
        [3, 2, 1, 2, 4, 2, 3, 2, 1]
    ],[
        [14, 14, 2, 4, 3, 1, 4, 2, 4],
        [14, 14, 4, 3, 1, 2, 1, 3, 3],
        [3, 4, 1, 5, 2, 4, 1, 2, 5],
        [5, 5, 4, 5, 2, 5, 5, 4, 4],
        [4, 1, 2, 3, 1, 2, 3, 4, 2],
        [4, 1, 4, 4, 2, 4, 1, 3, 4],
        [2, 4, 3, 3, 5, 5, 4, 1, 2],
        [1, 2, 1, 1, 3, 3, 1, 4, 1],
        [4, 1, 3, 2, 1, 2, 1, 5, 2],
        [3, 2, 1, 2, 4, 2, 3, 2, 1]
    ]))
        model = DQN(len(matrix_3_channels.shape), 161).to(DEVICE)
        print(matrix_3_channels.shape)
        input_tensor = torch.tensor(matrix_3_channels, dtype=torch.float).to(DEVICE)

        output = model(input_tensor)
        
        # find a way to return what the model thinks is the actual prediction of the index of the move that we should make
        max_value, max_index = torch.max(output, dim=0)
        print(float(max_value))
        print(int(max_index))

    def test_model_1d_matrix(self):
       
        matrix_1_channels = np.array(([
        [14, 14, 2, 4, 3, 1, 4, 2, 4],
        [14, 14, 4, 3, 1, 2, 1, 3, 3],
        [3, 4, 1, 5, 2, 4, 1, 2, 5],
        [5, 5, 4, 5, 2, 5, 5, 4, 4],
        [4, 1, 2, 3, 1, 2, 3, 4, 2],
        [4, 1, 4, 4, 2, 4, 1, 3, 4],
        [2, 4, 3, 3, 5, 5, 4, 1, 2],
        [1, 2, 1, 1, 3, 3, 1, 4, 1],
        [4, 1, 3, 2, 1, 2, 1, 5, 2],
        [3, 2, 1, 2, 4, 2, 3, 2, 1]
    ]))
        
        model = DQN(1, 161).to(DEVICE)
        print(matrix_1_channels.shape)
        input_tensor = torch.tensor(matrix_1_channels, dtype=torch.float).to(DEVICE)
        print(input_tensor.shape)

        output = model(input_tensor)
        
        # find a way to return what the model thinks is the actual prediction of the index of the move that we should make
        max_value, max_index = torch.max(output, dim=0)
        print(float(max_value))
        print(int(max_index))


if __name__ == '__main__':
    unittest.main()