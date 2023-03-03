import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import numpy as np
from tqdm import trange

import mushroom_rl
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import TRPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J


N_EVALS = 20

# Parameters
parameters = dict(
    env_id="MountainCarContinuous-v0",
    budget=50_000,
    n_eval_episodes=3,
    horizon=200,
    gamma=0.99,
    n_steps_per_fit=2048,
    ent_coeff=0.0,
    max_kl=0.01,
    lam=0.95,
    n_epochs_line_search=10,
    n_epochs_cg=15,
    cg_damping=0.1,
    cg_residual_tol=1e-10,
    policy_params=dict(
        std_0=1.,
        n_features=32,
        use_cuda=False,
    ),
    critic_params=dict(
        n_features=32,
        batch_size=64,
        learning_rate=3e-4,
    ),
    gpu=False, # Say whether you use GPU!
    mushroom_rl_version=mushroom_rl.__version__,
    gym_version=gym.__version__,
    numpy_version=np.__version__,
    python_version=sys.version,
    torch_version=torch.__version__
)
output_dir_name = "results/trpo/" # experiment folder
locals().update(parameters)  # load all the variables defined in parameters dict


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, required=True,
        help='Seed for the experiment')
    parser.add_argument('--env-id', '-e', type=str, default=env_id,
        help='Environment id (default: {})'.format(env_id))
    parser.add_argument('--n-eval-episodes', '-n', type=int, default=n_eval_episodes,
        help='Number of episodes for evaluation (default: {})'.format(n_eval_episodes))
    args = parser.parse_args()

    # gather parameters
    parameters['seed'] = args.seed
    parameters['env_id'] = args.env_id
    parameters['n_eval_episodes'] = args.n_eval_episodes
    locals().update(parameters)  # load all the variables defined in parameters dict
 
    # update directory for this run
    output_dir_name = os.path.join(output_dir_name, str(args.env_id), str(args.seed))
    os.makedirs(output_dir_name, exist_ok=True)

    # logging
    logger = Logger(TRPO.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + TRPO.__name__)

    # environment
    mdp = Gym(env_id, horizon, gamma)

    # agent
    critic_params = dict(
        network=Network,
        optimizer={
            'class': optim.Adam,
            'params': {'lr': parameters['critic_params']['learning_rate']}
        },
        loss=F.mse_loss,
        n_features=parameters['critic_params']['n_features'],
        batch_size=parameters['critic_params']['batch_size'],
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,)
    )
    
    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        **parameters['policy_params']
    )
    
    agent = TRPO(
        mdp.info, policy, critic_params=critic_params,
        ent_coeff=ent_coeff, max_kl=max_kl, lam=lam, 
        n_epochs_line_search=n_epochs_line_search, n_epochs_cg=n_epochs_cg,
        cg_damping=cg_damping, cg_residual_tol=cg_residual_tol
    )
    agent.set_logger(logger)

    # core
    core = Core(agent, mdp)

    # evaluation arrays
    timesteps = np.zeros(N_EVALS + 1, dtype=int)
    evaluations = np.zeros(N_EVALS + 1, dtype=float)

    # initial eval
    dataset = core.evaluate(n_episodes=n_eval_episodes, render=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)
    evaluations[0] = R

    # training
    eval_freq = budget // N_EVALS
    for it in trange(N_EVALS, leave=False):
        core.learn(n_steps=eval_freq, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_eval_episodes, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        logger.epoch_info(it+1, J=J, R=R, entropy=E)
        timesteps[it+1] = (it+1) * eval_freq
        evaluations[it+1] = R
        np.savez(os.path.join(output_dir_name, 'evaluations.npz'),
            timesteps=timesteps, evaluations=evaluations)

    # save agent and parameters
    parameters['save_path'] = os.path.join(output_dir_name, 'agent.msh')
    agent.save(parameters['save_path'])
    with open(os.path.join(output_dir_name, "parameters.json"), "w") as file:
        file.write(json.dumps(parameters))

    # evaluate and print results
    print(f'AdaStop Evaluation: {evaluations[-1]}')
