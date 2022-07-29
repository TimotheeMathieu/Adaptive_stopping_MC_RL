from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
import yaml
from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents
from torch import nn
from stable_baselines3 import PPO, DQN
import numpy as np
from compare_agents import Two_AgentsComparator


env_ctor, env_kwargs = gym_make, dict(id="CartPole-v1")

# Hyperparameters
with open(r'sb_hyperparams.yml') as file:
    hyperparams = yaml.load(file, Loader=yaml.FullLoader)

hyperparams['DQN']['policy_kwargs']= eval(hyperparams['DQN']['policy_kwargs'])

n_steps_ppo = hyperparams['PPO']['n_timesteps']
del hyperparams['PPO']['n_timesteps']

n_steps_a2c = hyperparams['DQN']['n_timesteps']
del hyperparams['DQN']['n_timesteps']

def linear_lr_schedule_ppo(progress_remaining: float) -> float:
    return progress_remaining * 0.001

hyperparams['PPO']['learning_rate'] = linear_lr_schedule_ppo

def linear_cr_schedule_ppo(progress_remaining: float) -> float:
    return progress_remaining * 0.2

hyperparams['PPO']['clip_range'] = linear_cr_schedule_ppo

# hyperparams['PPO']['device']='cpu'
# hyperparams['DQN']['device']="cpu"

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
        train_env=(env_ctor, env_kwargs),
        agent_name="DQN",
        init_kwargs=dict(algo_cls=DQN, **hyperparams['DQN']),
        fit_budget=n_steps_a2c,
        n_fit=15,
        parallelization="process",
    ))
    comparator.compare(manager1, manager2)
