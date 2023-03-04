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


# Parameters
parameters = dict(
    budget=1_000_000,
    policy= "MlpPolicy",
    env_id="HalfCheetah-v3",
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',
    use_sde=True,
    n_eval_episodes=50,
    eval_freq=100_000,
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
        tensorboard_log=output_dir_name, seed=seed)


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
    parser.add_argument('--eval-freq', '-f', type=int, default=eval_freq,
        help='Evaluation frequency (default: {})'.format(eval_freq))
    args = parser.parse_args()

    # gather parameters
    parameters['seed'] = args.seed
    parameters['env_id'] = args.env_id
    parameters['n_eval_episodes'] = args.n_eval_episodes
    parameters['eval_freq'] = args.eval_freq
    locals().update(parameters)  # load all the variables defined in parameters dict

    # create directory for this run
    output_dir_name = os.path.join(output_dir_name, str(args.env_id), str(args.seed))
    os.makedirs(output_dir_name, exist_ok=True)

    # set up agent
    agent = make_agent(args.seed, args.env_id)

    initial_eval = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval_episodes)[0]
    print(f"Initial Evaluation: {initial_eval}")

    callback = EvalCallback(
        agent.get_env(), n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
        log_path=output_dir_name, deterministic=False)

    # train
    agent.learn(budget, callback=callback)

    # prepare and save evaluation data
    n_evals = budget // eval_freq
    timesteps = np.zeros(n_evals + 1, dtype=int)
    evaluations = np.zeros(n_evals + 1, dtype=float)
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
    