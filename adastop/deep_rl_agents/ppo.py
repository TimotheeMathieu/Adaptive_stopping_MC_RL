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
import torch


N_EVALS = 20


# Parameters
parameters = dict(
    env_id="MountainCarContinuous-v0",
    layer_sizes=[64, 64],
    policy_net_fn="rlberry.agents.torch.utils.training.model_factory_from_env",
    batch_size=256,
    n_steps=2048,
    gamma=0.99,
    entr_coef=0.005,
    vf_coef=0.2,
    learning_rate=1e-4,
    eps_clip=0.1,
    k_epochs=10,
    gae_lambda=0.9,
    normalize_advantages=True,
    fit_budget=50_000,
    eval_horizon=500,
    n_eval_episodes=3,
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
    args = parser.parse_args()

    # gather parameters
    parameters['seed'] = args.seed
    parameters['env_id'] = args.env_id
    parameters['n_eval_episodes'] = args.n_eval_episodes
    locals().update(parameters)  # load all the variables defined in parameters dict
 
    # update directory for this run
    output_dir_name = os.path.join(output_dir_name, str(args.env_id), str(args.seed))
    os.makedirs(output_dir_name, exist_ok=True)

    # init agent
    policy_net_kwargs = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": layer_sizes,
        "reshape": False,
        "is_policy": True,
        "ctns_actions": True,
        "out_size": 2,
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
    agent.fit(0)

    # evaluation arrays
    timesteps = np.zeros(N_EVALS + 1, dtype=int)
    evaluations = np.zeros(N_EVALS + 1, dtype=float)

    # train
    n_evals, eval_freq = 0, int(fit_budget / N_EVALS)
    used_budget = 0
    evaluations[0] = np.mean(evaluate_agents([agent], n_simulations=n_eval_episodes, show=False).values)
    while used_budget < fit_budget:
        agent.fit(eval_freq)
        used_budget += eval_freq

        n_evals += 1
        timesteps[n_evals] = used_budget
        evaluations[n_evals] =np.mean(evaluate_agents([agent], n_simulations=n_eval_episodes, show=False).values)
        np.savez(os.path.join(output_dir_name, "evaluations.npz"), timesteps=timesteps, evaluations=evaluations)

    # save agent and parameters
    save_path = str(agent.save())
    parameters["save_path"] = save_path
    with open(os.path.join(output_dir_name, "parameters.json"), "w") as file:
        file.write(json.dumps(parameters))

    # evaluate and print results
    print(f'AdaStop Evaluation: {evaluations[-1]}')
