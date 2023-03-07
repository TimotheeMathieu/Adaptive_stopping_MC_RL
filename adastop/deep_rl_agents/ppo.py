import numpy as np
import json
import os
import sys

import rlberry
from rlberry.agents.torch import PPOAgent
from rlberry.envs import gym_make
from rlberry.manager import AgentManager, evaluate_agents
from rlberry.utils.torch import choose_device

import gym
from gym.wrappers import NormalizeObservation
import torch


# Parameters
parameters = dict(
    env_id="HalfCheetah-v3",
    layer_sizes=[32, 32],
    policy_net_fn="rlberry.agents.torch.utils.training.model_factory_from_env",
    batch_size=32,
    n_steps=2048,
    gamma=0.99,
    entr_coef=1e-5,
    vf_coef=0.5,
    learning_rate=3e-4,
    eps_clip=0.1,
    k_epochs=10,
    gae_lambda=0.95,
    normalize_advantages=True,
    normalize_rewards=True,
    fit_budget=1_000_000,
    eval_horizon=500,
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
def make_env(id=None):
    env = gym_make(id=id)
    env = NormalizeObservation(env)
    return env


# agent and environment definition
env_ctor = make_env
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
            learning_rate=learning_rate,
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            gae_lambda=gae_lambda,
            normalize_rewards=normalize_rewards,
            normalize_advantages=normalize_advantages, 
            policy_net_fn=policy_net_fn,
            policy_net_kwargs=policy_net_kwargs,
            device=choose_device("cuda:best" if gpu else "cpu"),
        ),
        fit_budget=fit_budget,
        eval_kwargs=dict(eval_horizon=eval_horizon),
        n_fit=n_fit,
        parallelization="process",
        mp_context="spawn",
        seed=args.seed,
        output_dir=output_dir_name # one folder for each agent
    )
    agent.fit(1)

    # evaluation arrays
    n_evals = fit_budget // eval_freq
    timesteps = np.zeros(n_evals + 1, dtype=int)
    evaluations = np.zeros(n_evals + 1, dtype=float)

    # train
    curr_eval_idx, used_budget = 0, 0
    evaluations[0] = np.mean(evaluate_agents([agent], n_simulations=n_eval_episodes, show=False).values)
    while used_budget < fit_budget:
        agent.fit(eval_freq)
        used_budget += eval_freq

        curr_eval_idx += 1
        timesteps[curr_eval_idx] = used_budget
        evaluations[curr_eval_idx] =np.mean(evaluate_agents([agent], n_simulations=n_eval_episodes, show=False).values)
        np.savez(os.path.join(output_dir_name, "evaluations.npz"), timesteps=timesteps, evaluations=evaluations)

    # save agent and parameters
    save_path = str(agent.save())
    parameters["save_path"] = save_path
    with open(os.path.join(output_dir_name, "parameters.json"), "w") as file:
        file.write(json.dumps(parameters))

    # evaluate and print results
    print(f'AdaStop Evaluation: {evaluations[-1]}')
