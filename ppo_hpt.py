import numpy as np
import json
import os
import sys

import rlberry
from rlberry.agents.torch import PPOAgent
from rlberry.envs import gym_make
from rlberry.manager import AgentManager, plot_writer_data
from rlberry.utils.torch import choose_device

import gym
import torch


class PPO(PPOAgent):
    @classmethod
    def sample_parameters(cls, trial):
        """
        Sample hyperparameters for hyperparam optimization using
        Optuna (https://optuna.org/)

        Note: only the kwargs sent to __init__ are optimized. Make sure to
        include in the Agent constructor all "optimizable" parameters.

        Parameters
        ----------
        trial: optuna.trial
        """
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 256])
        gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.9999])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        entr_coef = trial.suggest_loguniform("entr_coef", 1e-6, 1e-2)
        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "entr_coef": entr_coef,
        }



# Parameters
parameters = dict(
    env_id="HalfCheetah-v3",
    layer_sizes=[256, 256],
    policy_net_fn="rlberry.agents.torch.utils.training.model_factory_from_env",
    batch_size=64,
    n_steps=512,
    gamma=0.99,
    entr_coef=3e-4,
    vf_coef=0.5,
    learning_rate=2e-5,
    eps_clip=0.1,
    k_epochs=20,
    gae_lambda=0.9,
    normalize_advantages=False,
    normalize_rewards=True,
    fit_budget=1_000_000,
    eval_horizon=500,
    n_eval_episodes=50,
    eval_freq=100_000,
    n_fit=3,
    gpu=False, # Say whether you use GPU!
    rlberry_version=rlberry.__version__,
    gym_version=gym.__version__,
    numpy_version=np.__version__,
    python_version=sys.version,
    torch_version=torch.__version__
)
output_dir_name = "results/ppo/" # experiment folder
locals().update(parameters)  # load all the variables defined in parameters dict


# * Agent and environment definition
env_ctor = gym_make
env_kwargs = dict(id=env_id)


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
        PPO,
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

    agent.optimize_hyperparams(
        n_trials=100,
        n_fit=3,
        timeout=60,
    )
    print(agent.best_hyperparams)
    with open(os.path.join(output_dir_name, "best_hyperparams.json"), "w") as f:
        json.dump(agent.best_hyperparams, f)

    # 
    agent.fit()
    plot_writer_data(agent, tag="episode_rewards")