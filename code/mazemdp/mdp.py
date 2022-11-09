"""
Author: Olivier Sigaud
"""

import numpy as np

from mazemdp.toolbox import discreteProb



class Mdp:
    """
    defines a Markov Decision Process
    """

    def __init__(
        self,
        nb_states,
        action_space,
        start_distribution,
        transition_matrix,
        reward_matrix,
        gamma=0.9,
        terminal_states=None,
        has_state=True,
    ):
        self.nb_states = nb_states
        if terminal_states is None:
            terminal_states = []
        self.terminal_states = terminal_states
        self.action_space = action_space
        self.has_state = has_state
        self.timestep = 0
        self.P0 = start_distribution  # distribution used to draw the first state of the agent, used in method reset()
        self.P = transition_matrix
        self.r = reward_matrix
        self.gamma = gamma  # discount factor
        self.current_state = None

    def reset(
        self, uniform=False
    ):  # initializes an episode and returns the state of the agent
        # if uniform is set to False, the first state is drawn according to the P0 distribution,
        # else it is drawn from a uniform distribution over all the states except for walls

        if uniform:
            prob = np.ones(self.nb_states-1) / (self.nb_states-1)
            self.current_state = discreteProb(prob)
        else:
            self.current_state = discreteProb(self.P0)

        self.timestep = 0
        self.last_action_achieved = False
        return self.current_state

    def done(self):  # returns True if the episode is over
        if self.current_state in self.terminal_states:
            return True
        else:
            return False

    def step(self, u, deviation=0):  # performs a step forward in the environment,
        # if you want to add some noise to the reward, give a value to the deviation param
        # which represents the mean μ of the normal distribution used to draw the noise

        noise = deviation * np.random.randn()  # generate noise, useful for RTDP

        # r is the reward of the transition, you can add some noise to it
        reward = self.r[self.current_state, u] + noise
        # the state reached when performing action u from state x is sampled
        # according to the discrete distribution self.P[x,u,:]
        next_state = discreteProb(self.P[self.current_state, u, :])

        self.timestep += 1

        info = {
            "State transition probabilities": self.P[self.current_state, u, :],
            "reward's noise value": noise,
        }  # can be used when debugging

        self.current_state = next_state
        done = self.done()  # checks if the episode is over
        return [next_state, reward, done, info]
