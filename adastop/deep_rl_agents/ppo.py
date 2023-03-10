import json
import os
import sys

import rlberry
from rlberry.agents.torch import PPOAgent
from rlberry.envs import gym_make
from rlberry.manager import AgentManager, evaluate_agents, read_writer_data
from rlberry.utils.torch import choose_device

import gym
from gym import wrappers
import numpy as np
import pandas as pd
import torch


# Parameters
parameters = dict(
    env_id="HalfCheetah-v3",
    layer_sizes=[64, 64],
    policy_net_fn="rlberry.agents.torch.utils.training.model_factory_from_env",
    batch_size=64,
    n_steps=2048,
    gamma=0.99,
    target_kl=None,
    lr_schedule='linear',
    value_loss='avec',
    entr_coef=0.0,
    vf_coef=0.5,
    learning_rate=3e-4,
    clip_eps=0.2,
    k_epochs=10,
    gae_lambda=0.95,
    normalize_advantages=True,
    fit_budget=1_000_000,
    eval_horizon=1_000,
    n_eval_episodes=50,
    eval_freq=100_000,
    n_fit=1,
    gpu=False, # Say whether you use GPU!
    rlberry_version=rlberry.__version__,
    gym_version=gym.__version__,
    numpy_version=np.__version__,
    python_version=sys.version,
    torch_version=torch.__version__
)
output_dir_name = "results/ppo/" # experiment folder
locals().update(parameters)  # load all the variables defined in parameters dict


# make env with preprocessing
def get_make_env(eval=False):
    def make(id=None):
        env = gym_make(id=id)
        env = wrappers.FlattenObservation(env)
        env = wrappers.RecordEpisodeStatistics(env)
        env = wrappers.ClipAction(env)
        env = wrappers.NormalizeObservation(env)
        env = wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if not eval:
            env = wrappers.NormalizeReward(env, gamma=gamma)
            env = wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return make


# agent and environment definition
env_ctor = get_make_env()
eval_env_ctor = get_make_env(eval=True)
env_kwargs = dict(id=env_id)


# training and saving training data
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, required=True,
        help='Random seed for the experiment.')
    parser.add_argument('--env-id', '-e', type=str, default=env_id,
        help='Environment id (default: {})'.format(env_id))
    parser.add_argument('--n-eval-episodes', '-n', type=int, default=n_eval_episodes,
        help='Number of episodes for evaluation (default: {})'.format(n_eval_episodes))
    parser.add_argument('--eval-freq', '-f', type=int, default=eval_freq,
        help='Evaluation frequency (default: {})'.format(eval_freq))
    args = parser.parse_args()

    # gather parameters
    parameters['seed'] = args.seed
    parameters['env_id'] = args.env_id
    parameters['n_eval_episodes'] = args.n_eval_episodes
    parameters['eval_freq'] = args.eval_freq
    locals().update(parameters)  # load all the variables defined in parameters dict
 
    # update directory for this run
    output_dir_name = os.path.join(output_dir_name, str(args.seed))
    os.makedirs(output_dir_name, exist_ok=True)

    # init agent
    policy_net_kwargs = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": layer_sizes,
        "reshape": False,
        "is_policy": True,
        "ctns_actions": True,
        "out_size": env_ctor(**env_kwargs).action_space.shape[0],
    }

    agent = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        init_kwargs=dict(
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            entr_coef=entr_coef,
            vf_coef=vf_coef,
            value_loss=value_loss,
            learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            clip_eps=clip_eps,
            k_epochs=k_epochs,
            gae_lambda=gae_lambda,
            normalize_advantages=normalize_advantages, 
            policy_net_fn=policy_net_fn,
            policy_net_kwargs=policy_net_kwargs,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            eval_horizon=eval_horizon,
            device=choose_device("cuda:best" if gpu else "cpu"),
        ),
        eval_env=(eval_env_ctor, env_kwargs),
        fit_budget=fit_budget,
        eval_kwargs=dict(eval_horizon=eval_horizon),
        n_fit=n_fit,
        parallelization="process",
        mp_context="spawn",
        seed=args.seed,
        output_dir=output_dir_name # one folder for each agent
    )
    
    # train
    agent.fit(fit_budget)

    # get logged data
    data = read_writer_data(output_dir_name, tag='evaluation')
    data.to_csv(os.path.join(output_dir_name, "data.csv"))

    evals = data[data['tag'] == 'evaluation']
    timesteps = evals['global_step'].to_numpy()
    evaluations = evals['value'].to_numpy()
    np.savez(os.path.join(output_dir_name, "evaluations.npz"), timesteps=timesteps, evaluations=evaluations)

    # save agent and parameters
    save_path = str(agent.save())
    parameters["save_path"] = save_path
    with open(os.path.join(output_dir_name, "parameters.json"), "w") as file:
        file.write(json.dumps(parameters))

    # evaluate and print results
    print(f'AdaStop Evaluation: {evaluations[-1]}')