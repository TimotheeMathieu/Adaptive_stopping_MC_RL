from rlberry.agents import Agent
from rlberry.envs import Model

import rlberry.spaces as spaces
import numpy as np


class DummyEnv(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        self.n_arms = 2
        self.action_space = spaces.Discrete(1)

    def step(self, action):
        pass

    def reset(self):
        return 0

class RandomAgent(Agent):
    def __init__(self, env, drift=0, std = 1, **kwargs):
        Agent.__init__(self, env)
        self.drift = drift
        self.std = std
        if "type" in kwargs.keys():
            self.type = kwargs["type"]
        else:
            self.type = "normal"
        self.kwargs = kwargs

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):
        if self.type == "normal":
            noise = self.rng.normal(size=n_simulations)*self.std
        elif self.type == "student":
            if "df" in self.kwargs.keys():
                df = self.kwargs["df"]
            else:
                df = 2.
            noise = self.rng.standard_t(df, size=n_simulations)
        return self.drift + noise

class MixtureAgent(Agent): 
    def __init__(self, env, means=[0], stds=[1], prob_mixture = [1], **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.means = np.array(means)
        self.prob_mixture = np.array(prob_mixture)
        self.stds = np.array(stds)
        if "type" in kwargs.keys():
            self.type = kwargs["type"]
        else:
            self.type = "normal"

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):

        if self.type == "normal":
            noise = self.rng.normal(size=n_simulations)
        elif self.type == "student":
            if "df" in self.kwargs.keys():
                df = self.kwargs["df"]
            else:
                df = 2.
            noise = self.rng.standard_t(df, size=n_simulations)

        idxs = self.rng.choice(np.arange(len(self.means)), size=n_simulations, p=self.prob_mixture)
        ret = self.means[idxs] + noise*self.stds[idxs]
        return ret


#TODO check that comparator is using n_simulations
class MixtureGaussianAgent(Agent):
    def __init__(self, env, means=[0], stds=[1], prob_mixture = [1], **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.means = np.array(means)
        self.prob_mixture = np.array(prob_mixture)
        self.stds = np.array(stds)

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):
        idxs = self.rng.choice(np.arange(len(self.means)), size=n_simulations, p=self.prob_mixture)
        ret = self.means[idxs] + self.rng.normal(size=n_simulations)*self.stds[idxs]
        return ret


def make_same_agents(diff_means, probas = [0.5, 0.5]):
    manager1 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0, 0+diff_means], "stds": [0.1, 0.1], "prob_mixture": probas},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )
    manager2 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0, 0 + diff_means], "stds": [0.1, 0.1], "prob_mixture": probas},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )
    return manager1, manager2


def make_different_agents(mus, probas = [0.5, 0.5]):
    manager1 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": mus, "stds": [0.1, 0.1], "prob_mixture": probas},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )
    manager2 = (
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": 0, "std": 0.1},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )
    return manager1, manager2



def create_agents(agent_name, agent_label, **kwargs):
    if agent_name == "mixture":
        assert "mus" in kwargs.keys() and "probas" in kwargs.keys()
        manager =  (
                        MixtureAgent,
                        dict(
                            train_env=(DummyEnv, {}),
                            init_kwargs={"means": kwargs["mus"], "stds": [0.1, 0.1], "prob_mixture": kwargs["probas"]},
                            fit_budget=1,
                            agent_name=agent_label,
                        ),
                    )
    elif agent_name == "single":
        init_kwargs = dict(type = kwargs["type"], drift = kwargs["drift"])
        if kwargs["type"] == "student":
            init_kwargs["df"] = kwargs["df"]
        manager =  (
                        RandomAgent,
                        dict(
                            train_env=(DummyEnv, {}),
                            init_kwargs=init_kwargs,
                            fit_budget=1,
                            agent_name=agent_label,
                        ),
                    )   

    return manager

# def exp1(diff_means):
#     mus = [0-diff_means/2, 0+diff_means/2]
#     return make_different_agents(mus = mus)

# def exp2(diff_means):
#     mus = [0, diff_means]
#     return make_different_agents(mus = mus)

# def exp3(df):
#     manager1 = (
#         RandomAgent,
#         dict(
#             train_env=(DummyEnv, {}),
#             init_kwargs={"drift": 0, "df": df, "type": "student"},
#             fit_budget=1,
#             agent_name="Agent1",
#         ),
#     )
#     manager2 = (
#         RandomAgent,
#         dict(
#             train_env=(DummyEnv, {}),
#             init_kwargs={"drift": 0, "std": 1.},
#             fit_budget=1,
#             agent_name="Agent2",
#         ),
#     )
#     return manager1, manager2
