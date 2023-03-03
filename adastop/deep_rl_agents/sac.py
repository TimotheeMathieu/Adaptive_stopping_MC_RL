import numpy as np
import json
import os
import sys

import gym
import torch

import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


N_EVALS = 20


# Parameters
parameters = dict(
    budget=50_000,
    policy= "MlpPolicy",
    env_id="MountainCarContinuous-v0",
    learning_rate=3e-4,
    buffer_size=50_000,
    learning_starts=0,
    batch_size=512,
    tau=0.01,
    gamma=0.9999,
    train_freq=32,
    gradient_steps=1,
    ent_coef=0.1,
    use_sde=True,
    policy_kwargs=dict(
        log_std_init=-3.67,
        net_arch=[64, 64],
    ),
    n_eval_episodes=3,
    gpu=False, # Say whether you use GPU!
    stable_baselines3_version=stable_baselines3.__version__,
    gym_version=gym.__version__,
    numpy_version=np.__version__,
    python_version=sys.version,
    torch_version=torch.__version__
)
output_dir_name = "results/sac/" # experiment folder
locals().update(parameters)  # load all the variables defined in parameters dict


def make_agent(seed, env_id):
    return SAC(policy, env_id, learning_rate=learning_rate, 
        buffer_size=buffer_size, learning_starts=learning_starts,
        batch_size=batch_size, tau=tau, gamma=gamma, train_freq=train_freq,
        gradient_steps=gradient_steps, ent_coef=ent_coef, use_sde=use_sde,
        policy_kwargs=policy_kwargs, tensorboard_log=output_dir_name,
        seed=seed)


# * Training and saving training data
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, required=True,
        help='Random seed for the experiment.')
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

    # create directory for this run
    output_dir_name = os.path.join(output_dir_name, str(args.env_id), str(args.seed))
    os.makedirs(output_dir_name, exist_ok=True)

    # set up agent
    agent = make_agent(args.seed, args.env_id)

    initial_eval = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval_episodes)[0]
    print(f"Initial Evaluation: {initial_eval}")

    callback = EvalCallback(
        agent.get_env(), n_eval_episodes=n_eval_episodes, eval_freq=budget // N_EVALS,
        log_path=output_dir_name, deterministic=False)

    # train
    agent.learn(budget, callback=callback)

    # prepare and save evaluation data
    timesteps = np.zeros(N_EVALS + 1, dtype=int)
    evaluations = np.zeros(N_EVALS + 1, dtype=float)
    evaluations[0] = initial_eval

    callback_output = np.load(os.path.join(output_dir_name, "evaluations.npz"))
    timesteps[1:] = callback_output["timesteps"]
    evaluations[1:] = np.mean(callback_output["results"], axis=-1)

    np.savez(os.path.join(output_dir_name, "evaluations.npz"),
        timesteps=timesteps, evaluations=evaluations)
    
    # save agent and parameters
    parameters["save_path"] = os.path.join(output_dir_name, "sac.zip")
    agent.save(parameters["save_path"])
    with open(os.path.join(output_dir_name, "parameters.json"), "w") as file:
        file.write(json.dumps(parameters))

    # evaluate and print results
    r_mean, r_std = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval_episodes)
    print(f"AdaStop Evaluation: {r_mean}")
    