import numpy as np
from rlberry.agents import AgentWithSimplePolicy
import rlberry
from gym import spaces

logger = rlberry.logger


class QLAgent(AgentWithSimplePolicy):
    name = "QL"

    def __init__(
        self,
        env,
        gamma=0.99,
        alpha=0.1,
        epsilon=0.3,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.eps = epsilon
        self.alpha = alpha
        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)
        # initialize
        self.reset()

    def reset(self, **kwargs):
        S = self.env.observation_space.n
        A = self.env.action_space.n
        self.Q = np.zeros((S,A))

    def policy(self, observation):
        """Recommended policy."""
        s= observation
        return self.Q[s].argmax()

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.
        Parameters
        ----------
        budget: int
            number of Q updates.
        """
        del kwargs
        s = self.env.reset()
        for i in range(budget):
            if np.random.random() <= self.eps:
                a = np.random.choice(self.env.action_space.n)
            else:
                a = self.Q[s].argmax()
            snext, r , done, _ = self.env.step(a)
            if self.writer is not None:
                self.writer.add_scalar("reward", r, i)
            self.Q[s,a] = self.Q[s,a] + self.alpha * (r + self.gamma * np.amax(self.Q[snext]) - self.Q[s,a])
            s = snext
            if done:
                s = self.env.reset()


class SARSAAgent(AgentWithSimplePolicy):
    name = "SARSA"

    def __init__(
        self,
        env,
        gamma=0.99,
        alpha=0.1,
        epsilon=0.3,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.eps = epsilon
        self.alpha = alpha
        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)
        # initialize
        self.reset()

    def reset(self, **kwargs):
        S = self.env.observation_space.n
        A = self.env.action_space.n
        self.Q = np.zeros((S,A))

    def policy(self, observation):
        """Recommended policy."""
        s= observation
        return self.Q[s].argmax()


    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.
        Parameters
        ----------
        budget: int
            number of Q updates.
        """
        del kwargs
        s = self.env.reset()
        for i in range(budget):

            if np.random.random() <= self.eps:
                a = np.random.choice(self.env.action_space.n)
            else:
                a = self.Q[s].argmax()


            snext, r , done, _ = self.env.step(a)

            if np.random.random() <= self.eps:
                anext = np.random.choice(self.env.action_space.n)
            else:
                anext = self.Q[snext].argmax()


            if self.writer is not None:
                self.writer.add_scalar("reward", r, i)
            self.Q[s,a] = self.Q[s,a] + self.alpha * (r + self.gamma * self.Q[snext, anext] - self.Q[s,a])
            s = snext
            if done:
                s = self.env.reset()

if __name__ == "__main__":

    from gym_stochastic import TwentyOneWithDice

    env = TwentyOneWithDice()
    ql = QLAgent(env)
    sarsa = SARSAAgent(env)
    ql.fit(100)
    sarsa.fit(100)
