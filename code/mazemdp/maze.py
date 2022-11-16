"""
Author: Olivier Sigaud
"""

import random

import numpy as np

from mazemdp.mdp import Mdp
from mazemdp.toolbox import E, N, S, W
import gym

def check_navigability(mdp):
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    stop = False

    while not stop:
        v_old = v.copy()

        for x in range(mdp.nb_states):  # for each state x
            # Compute the value of state x for each action u of the MDP action space
            if x not in mdp.terminal_states:
                v_temp = []
                for u in range(mdp.action_space.n):
                    # Process sum of the values of the neighbouring states
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ = summ + mdp.P[x, u, y] * v_old[y]
                    v_temp.append(mdp.r[x, u] + summ)

                # Select the highest state value among those computed
                v[x] = np.max(v_temp)

        # Test if convergence has been reached
        if (np.linalg.norm(v - v_old)) < 0.01:
            stop = True

    # We should reach terminal states from any starting point
    reachable = mdp.nb_states - np.count_nonzero(v) == len(mdp.terminal_states)
    return reachable


def build_maze(width, height, walls, hit=False):
    ts = height * width - 1 - len(walls)
    maze = Maze(
        width, height, hit, walls=walls, last_states=[ts]
    )  # Markov Decision Process definition

    # The MDP has one state more than the Maze (the final state
    # outside of the maze)
    return maze.mdp, maze.nb_states + 1


def create_random_maze(width, height, ratio, seed, hit=False):
    size = width * height
    n_walls = round(ratio * size)

    stop = False
    mdp = None
    random.seed(seed)
    # the loop below is used to check that the maze has a solution
    # if one of the values after check_navigability is null, then another maze should be produced
    while not stop:
        walls = random.sample(range(size), int(n_walls))

        mdp, nb_states= build_maze(width, height, walls, hit=hit)
        stop = check_navigability(mdp)
    return mdp, nb_states


class Maze:  # describes a maze-like environment
    def __init__(
        self,
        width,
        height,
        hit=False,
        walls=None,
        action_list=None,
        nb_actions=4,
        gamma=0.9,
        start_states=None,
        last_states=None,
    ):
        """
        :param width: Int number defining the maze width
        :param height: Int number defining the maze height
        :param walls: List of the states that represent walls in our maze environment
        :param action_list: List of possible actions
        :param nb_actions: used when action_list is empty, by default there are 4 of them (go north, south, eat or west)
        :param gamma: Discount factor of the mdp
        :param timeout: Defines the length of an episode (max timestep) --see done() function
        :param start_states: List defining the states where the agent can be at the beginning of an episode
        :param last_states: List defining the states corresponding to the step before the end of an episode
        """
        self.width = width
        self.height = height
        self.cells = np.zeros((width, height), int)
        self.nb_states = height * width - len(walls)

        self.gamma = gamma

        if walls is None:
            walls = []

        if action_list is None:
            action_list = []

        if start_states is None:
            start_states = [0]

        self.last_states = last_states
        if self.last_states is None:
            self.last_states = []

        self.well = self.nb_states  # all the final states' transitions go there

        self.walls = walls
        self.size = width * height

        self.state_width = []
        self.state_height = []
        # ##################### State Space ######################
        self.init_states(width, height, walls)

        # ##################### Action Space ######################
        self.action_space = gym.spaces.Discrete(4)

        # ##################### Distribution Over Initial States ######################

        start_distribution = np.zeros(
            self.nb_states
        )  # distribution over initial states

        # supposed to be uniform
        for state in start_states:
            start_distribution[state] = 1.0 / len(start_states)

        # ##################### Transition Matrix ######################
        transition_matrix = self.init_transitions(hit)

        if hit:
            reward_matrix = self.reward_hit_walls(transition_matrix)
        else:
            reward_matrix = self.simple_reward(transition_matrix)


        self.mdp = Mdp(
            self.nb_states + 1,
            self.action_space,
            start_distribution,
            transition_matrix,
            reward_matrix,
            gamma=self.gamma,
            terminal_states=[self.nb_states],
        )

    def init_states(self, width, height, walls):
        state = 0
        cell = 0
        for i in range(width):
            for j in range(height):
                if cell not in walls:
                    self.cells[i][j] = state
                    state = state + 1
                    self.state_width.append(i)
                    self.state_height.append(j)
                else:
                    self.cells[i][j] = -1
                cell = cell + 1

        assert self.nb_states == state, "maze init: error in the number of states"

    def init_transitions(self, hit):
        """
        Init the transition matrix
        a "well" state is added that only the terminal states can get into
        """

        transition_matrix = np.empty(
            (self.nb_states + 1, self.action_space.n, self.nb_states + 1)
        )

        transition_matrix[:, N, :] = np.zeros((self.nb_states + 1, self.nb_states + 1))
        transition_matrix[:, S, :] = np.zeros((self.nb_states + 1, self.nb_states + 1))
        transition_matrix[:, E, :] = np.zeros((self.nb_states + 1, self.nb_states + 1))
        transition_matrix[:, W, :] = np.zeros((self.nb_states + 1, self.nb_states + 1))

        for i in range(self.width):
            for j in range(self.height):
                state = self.cells[i][j]
                if not state == -1:

                    # Transition Matrix when going north (no state change if highest cells or cells under a wall)
                    if j == 0 or self.cells[i][j - 1] == -1:
                        transition_matrix[state][N][state] = 1.0
                    else:  # it goes up
                        transition_matrix[state][N][self.cells[i][j - 1]] = 1.0

                    # Transition Matrix when going south (no state change if lowest cells or cells above a wall)
                    if j == self.height - 1 or self.cells[i][j + 1] == -1:
                        transition_matrix[state][S][state] = 1.0
                    else:  # it goes down
                        transition_matrix[state][S][self.cells[i][j + 1]] = 1.0

                    # Transition Matrix when going east (no state change if left cells or on the left side of a wall)
                    if i == self.width - 1 or self.cells[i + 1][j] == -1:
                        transition_matrix[state][E][state] = 1.0
                    else:  # it goes left
                        transition_matrix[state][E][self.cells[i + 1][j]] = 1.0

                    # Transition Matrix when going west (no state change if right cells or on the right side of a wall)
                    if i == 0 or self.cells[i - 1][j] == -1:
                        transition_matrix[state][W][state] = 1.0
                    else:  # it goes right
                        transition_matrix[state][W][self.cells[i - 1][j]] = 1.0

        # Transition Matrix of terminal states
        for s in self.last_states:
            transition_matrix[s, :, :] = 0
            transition_matrix[s, :, self.well] = 1

        return transition_matrix

        # self.mdp = MyMdp(self.nb_states, self.action_space, start_distribution, transition_matrix, reward_matrix,
        #                plotter, proba_action=0.5, gamma=gamma, terminal_states=terminal_states, timeout=timeout)

    # --------------------------------- Reward Matrix ---------------------------------
    def simple_reward(self, transition_matrix: np.array):
        reward_matrix = np.zeros((self.nb_states, self.action_space.n))
        for from_state, action in zip(*np.nonzero(transition_matrix[:, :, self.well])):
            reward_matrix[from_state, action] = 1.0
        return reward_matrix

    # --------------------------------- Reward Matrix ---------------------------------
    def reward_hit_walls(self, transition_matrix: np.array):
        # Get the for reaching a final state
        reward_matrix = self.simple_reward(transition_matrix)

        # Add negative rewards for hiting a wall
        for i in range(self.width):
            for j in range(self.height):
                state = self.cells[i][j]
                if not state == -1:

                    # Reward Matrix when going north
                    if (
                        j == 0 or self.cells[i][j - 1] == -1
                    ):  # highest cells + cells under a wall
                        reward_matrix[state, N] = -0.5

                    # Reward Matrix when going south
                    if (
                        j == self.height - 1 or self.cells[i][j + 1] == -1
                    ):  # lowest cells + cells above a wall
                        reward_matrix[state, S] = -0.5

                    # Reward Matrix when going east
                    if (
                        i == self.width - 1 or self.cells[i + 1][j] == -1
                    ):  # cells on the left + left side of a wall
                        reward_matrix[state, E] = -0.5

                    # Reward Matrix when going west
                    if (
                        i == 0 or self.cells[i - 1][j] == -1
                    ):  # cells on the right + right side of a wall
                        reward_matrix[state, W] = -0.5

        return reward_matrix
