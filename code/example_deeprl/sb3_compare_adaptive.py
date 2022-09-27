from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
import yaml
from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents
from torch import nn
from stable_baselines3 import PPO, A2C
import numpy as np
from compare_agents import Two_AgentsComparator

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: gym_make("Pendulum-v1")])
env_ctor_a2c, env_kwargs_a2c = VecNormalize, dict(venv=env)


env_ctor, env_kwargs = gym_make, dict(id="Pendulum-v1")



# Hyperparameters
with open(r'sb_hyperparams.yml') as file:
    hyperparams = yaml.load(file, Loader=yaml.FullLoader)

hyperparams['A2C']['policy_kwargs']= eval(hyperparams['A2C']['policy_kwargs'])

n_steps_ppo = hyperparams['PPO']['n_timesteps']
del hyperparams['PPO']['n_timesteps']

n_steps_a2c = hyperparams['A2C']['n_timesteps']
del hyperparams['A2C']['n_timesteps']


def linear_lr_schedule_a2c(progress_remaining: float) -> float:
    return progress_remaining * 7e-4

hyperparams['A2C']['learning_rate']=linear_lr_schedule_a2c

n = 4
K = 5
alpha = 0.05

comparator = Two_AgentsComparator(n, K, alpha, n_evaluations=30)

if __name__ == "__main__":
    
    manager1 = (
        StableBaselinesAgent,
        dict(
        train_env = (env_ctor, env_kwargs),
        agent_name="PPO",
        init_kwargs=dict(algo_cls=PPO, **hyperparams['PPO']),
        fit_budget=n_steps_ppo,
        n_fit=15,
        parallelization="process",
    ))
    manager2 = (
        StableBaselinesAgent,
        dict(
        train_env=(env_ctor_a2c, env_kwargs_a2c),
        agent_name="A2C",
        init_kwargs=dict(algo_cls=A2C, **hyperparams['A2C']),
        fit_budget=n_steps_a2c,
        n_fit=15,
        parallelization="process",
    ))
    comparator.compare(manager1, manager2)
