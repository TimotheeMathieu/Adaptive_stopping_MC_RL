"""
Simple Maze MDP
"""
import gym
from gym import spaces

from mazemdp import create_random_maze
from mazemdp.maze import build_maze

class MazeMDPEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array", "human"], "video.frames_per_second": 5}

    def __init__(self, width=6, height=6, frac=0.3, seed = 42,**kwargs):

        self.mdp, nb_states = create_random_maze(width, height, frac, seed)
        self.nb_states = nb_states
        self.observation_space = spaces.Discrete(nb_states)
        self.action_space = spaces.Discrete(4)
        self.terminal_states = [nb_states - 1]
        self.P = self.mdp.P
        self.gamma = self.mdp.gamma
        self.r = self.mdp.r

    def step(self, action):
        [s,r,done,infos] = self.mdp.step(action)
        return s,r,done,infos

    def reset(self, **kwargs):
        return self.mdp.reset(**kwargs)
