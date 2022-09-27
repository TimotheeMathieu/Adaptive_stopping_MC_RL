from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
import yaml
from rlberry.manager import AgentManager
from torch import nn
from multiple_managers import MultipleManagers
from stable_baselines3 import PPO, DQN
import numpy as np

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

hyperparams['PPO']['device']='cpu'
hyperparams['DQN']['device']="cpu"

# Comparison

dirnames = ["rlberry_data/PPO_sb", "rlberry_data/DQN_sb"]
if __name__ == "__main__":
    
    manager1 = AgentManager(
        StableBaselinesAgent,
        (env_ctor, env_kwargs),
        agent_name="PPO",
        init_kwargs=dict(algo_cls=PPO, **hyperparams['PPO']),
        fit_budget=n_steps_ppo,
        n_fit=15,
        parallelization="process",
        output_dir=dirnames[0],
    )
    manager2 = AgentManager(
        StableBaselinesAgent,
        (env_ctor, env_kwargs),
        agent_name="DQN",
        init_kwargs=dict(algo_cls=DQN, **hyperparams['DQN']),
        fit_budget=n_steps_a2c,
        n_fit=15,
        parallelization="process",
        output_dir=dirnames[1],
    )
    # manager1.fit()
    # manager1.save()
    # manager2.fit()
    # manager2.save()
    multimanager = MultipleManagers()

    multimanager.append(manager1)
    multimanager.append(manager2)
    multimanager.load(dirnames)
    multimanager.pilot_study(using_fits=np.arange(5))


# On 5 fits for each agent, I got that the necessary number of seeds is 18.
