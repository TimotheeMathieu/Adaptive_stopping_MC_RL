from rlberry.agents import Agent
from rlberry.envs import Model
from rlberry.seeding import Seeder
from rlberry.envs.bandits import  Bandit
from scipy import stats
import rlberry.spaces as spaces
import numpy as np
from rlberry.manager import AgentManager
from rlberry.seeding import Seeder
from adastop import MultipleAgentsComparator
from joblib import Parallel, delayed

class RlberryComparator(MultipleAgentsComparator):
    def __init__(self, n_evaluations = 100, **kwargs):
        MultipleAgentsComparator.__init__(self,**kwargs)
        self.n_evaluations = n_evaluations
        
    def compare(self, managers, verbose=False):
        Z = [np.array([]) for _ in managers]
        self.n = np.array([self.n]*len(managers))
        # spawn independent seeds, one for each fit and one for the comparator.
        seeder = Seeder(self.rng.randint(10000))
        seeders = seeder.spawn(len(managers) * self.K + 1)
        self.rng = seeders[-1].rng

        for k in range(self.K):
            Z = self._fit(managers, Z, k, seeders)

            self.partial_compare({self.agent_names[i] : Z[i] for i in range(len(managers))}, verbose)
            decisions = np.array(list(self.decisions.values()))
            if np.all([d in ["smaller", "larger", "equal"] for d in decisions]):
                break
        

        return self.decisions
    def _fit(self, managers, Z, k, seeders):
        """
        fit rlberry agents.
        """
        agent_classes = [manager[0] for manager in managers]
        kwargs_list = [manager[1] for manager in managers]
        for kwarg in kwargs_list:
            kwarg["n_fit"] = self.n[0]
        managers_in = []
        for i in range(len(agent_classes)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                agent_class = agent_classes[i]
                kwargs = kwargs_list[i]
                seeder = seeders[i]
                managers_in.append(AgentManager(agent_class, **kwargs, seed=seeder))
        if self.agent_names is None:
            self.agent_names = [manager.agent_name for manager in managers_in]

        # For now, paralellize only training because _get_evals not pickleable
        managers_in = Parallel(n_jobs=1, backend="multiprocessing")(
            delayed(_fit_agent)(manager) for manager in managers_in
        )

        idz = 0
        for i in range(len(agent_classes)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                Z[i] = np.hstack([Z[i], self._get_evals(managers_in[idz])])
                idz += 1
        return Z

    def _get_evals(self, manager):
        """
        Can be overwritten for alternative evaluation function.
        """
        eval_values = []
        for idx in range(self.n[0]):
            eval_values.append(
                np.mean(manager.eval_agents(self.n_evaluations, agent_id=idx))
            )
        return eval_values

def _fit_agent(manager):
    manager.fit()
    return manager
    

class DummyEnv(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        self.n_arms = 2
        self.action_space = spaces.Discrete(1)

    def step(self, action):
        pass

    def reset(self, seed):
        return 0

class RandomAgent(Agent):
    def __init__(self, env, drift=0, std = 1, **kwargs):
        Agent.__init__(self, env)
        
        if "type" in kwargs.keys():
            self.type = kwargs["type"]
        else:
            self.type = "normal"
            
        if self.type == "normal":
            law = stats.norm(loc = drift, scale=std)
        elif self.type == "student":
            if "df" in kwargs.keys():
                df = kwargs["df"]
            else:
                df = 2.
            law = stats.t(df, loc = drift)

        self.bandit = Bandit([law])
        self.bandit.seeder = Seeder(self.rng.integers(low=0, high=10000))
        
    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations = 1, **kwargs):
        # We will use only one simulation because we simulate directly law of empirical mean.
        return self.bandit.step(0)[1]

class MixtureLaw(): 
    def __init__(self, means=[0], stds=[1], prob_mixture = [1], **kwargs):
        self.means = np.array(means)
        self.prob_mixture = np.array(prob_mixture)
        self.stds = np.array(stds)
        if "type" in kwargs.keys():
            self.type = kwargs["type"]
        else:
            self.type = "normal"

    def rvs(self, size, random_state):
        if self.type == "normal":
            noise = random_state.normal(size=size)
        elif self.type == "student":
            if "df" in self.kwargs.keys():
                df = self.kwargs["df"]
            else:
                df = 2.
            noise = random_state.standard_t(df, size=size)
        idxs = random_state.choice(np.arange(len(self.means)), size=size, p=self.prob_mixture)
        ret = self.means[idxs] + noise*self.stds[idxs]
        return ret
    def mean(self):
        return np.sum(self.means*self.prob_mixture)

#TODO check that comparator is using n_simulations
class MixtureGaussianAgent(Agent):
    def __init__(self, env, means=[0], stds=[1], prob_mixture = [1], **kwargs):
        Agent.__init__(self, env, **kwargs)
        law = MixtureLaw(means, stds, prob_mixture)
        self.bandit = Bandit([law])

        self.bandit.seeder = Seeder(self.rng.integers(low=0, high=10000))

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations = 1, **kwargs):
        # We will use only one simulation because we simulate directly law of empirical mean.
        return self.bandit.step(0)[1]
    

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
            MixtureGaussianAgent,
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

