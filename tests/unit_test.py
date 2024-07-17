import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from gym_match3.envs.game import (Board,
                                  RandomBoard,
                                  CustomBoard,
                                  Point,
                                  Cell,
                                  AbstractSearcher,
                                  MatchesSearcher,
                                  Filler,
                                  Game,
                                  MovesSearcher,
                                  OutOfBoardError,
                                  ImmovableShapeError)
from gym_match3.envs.levels import (Match3Levels,
                                    Level)

class TestBoard(unittest.TestCase):

    def setUp(self):
        self.board = Board(columns=2, rows=2, n_shapes=3)
        board = np.array([
            [0, 1], 
            [2, 0]
        ])
        self.board.set_board(board)
    
    def test_board(self):
        s = self.board.board_size
        print(s)
        self.assertEqual(s, (2, 2))

    def test_board_type(self):
        print('work1')
        print(self.board.board_size)
        print('work2')
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()