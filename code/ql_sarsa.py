import numpy as np
from rlberry.agents import AgentWithSimplePolicy
import rlberry
from gym import spaces
from scipy.special import softmax

logger = rlberry.logger


class QLAgent(AgentWithSimplePolicy):
    name = "QL"

    def __init__(
        self,
        env,
        gamma=0.99,
        alpha=0.1,
        epsilon=None,
        tau=None,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.eps = epsilon
        self.tau = tau
        self.alpha = alpha
        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)
        assert (self.eps is not None and self.tau is None) or (self.tau is not None and self.eps is None)
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

    def get_action(self, s):
        if self.eps:
            if np.random.random() <= self.eps:
                a = np.random.choice(self.env.action_space.n)
            else:
                a = self.Q[s].argmax()
        else:
            a = np.random.choice(self.env.action_space.n, p = softmax(self.tau* self.Q[s]))
        return a

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
        cumul_r = 0
        for i in range(budget):
            a = self.get_action(s)
            snext, r , done, _ = self.env.step(a)
            cumul_r += r
            if self.writer is not None:
                self.writer.add_scalar("cumul_reward", cumul_r, i)

            if done:
                self.Q[s,a] = r
            else:
                self.Q[s,a] = self.Q[s,a] + self.alpha * (r + self.gamma * np.amax(self.Q[snext]) - self.Q[s,a])
            s = snext
            if done:
                s = self.env.reset()
                cumul_r = 0

class SARSAAgent(AgentWithSimplePolicy):
    name = "SARSA"

    def __init__(
        self,
        env,
        gamma=0.99,
        alpha=0.1,
        epsilon=None,
        tau=None,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.eps = epsilon
        self.tau = tau
        self.alpha = alpha
        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)
        assert (self.eps is not None and self.tau is None) or (self.tau is not None and self.eps is None)
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

    def get_action(self, s):
        if self.eps:
            if np.random.random() <= self.eps:
                a = np.random.choice(self.env.action_space.n)
            else:
                a = self.Q[s].argmax()
        else:
            a = np.random.choice(self.env.action_space.n, p = softmax(self.tau* self.Q[s]))
        return a



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
        cumul_r = 0
        for i in range(budget):
            a = self.get_action(s)
            snext, r , done, _ = self.env.step(a)
            cumul_r += r
            anext = self.get_action(snext)

            if self.writer is not None:
                self.writer.add_scalar("cumul_reward", cumul_r, i)

            if done:
                self.Q[s,a] = r
            else:
                self.Q[s,a] = self.Q[s,a] + self.alpha * (r + self.gamma * self.Q[snext, anext] - self.Q[s,a])
            s = snext
            if done:
                s = self.env.reset()
                cumul_r = 0

if __name__ == "__main__":

    from gym_stochastic import TwentyOneWithDice
    from rlberry.manager import AgentManager, evaluate_agents

    env = TwentyOneWithDice
    # ql = QLAgent(env)
    # sarsa = SARSAAgent(env)

    ql_manager = AgentManager(
        QLAgent,
        (env, {}),
        init_kwargs = dict(epsilon=0.3),
        fit_budget=100,
        eval_kwargs=dict(eval_horizon=100),
        n_fit=2,)

    ql_manager.fit()

    sarsa_manager = AgentManager(
        SARSAAgent,
        (env, {}),
        init_kwargs = dict(epsilon=0.3),
        fit_budget=100,
        eval_kwargs=dict(eval_horizon=100),
        n_fit=2,)

    sarsa_manager.fit()


    output = evaluate_agents([ql_manager, sarsa_manager], n_simulations=100)
