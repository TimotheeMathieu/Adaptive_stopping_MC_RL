import anomalous_gym
import numpy as np
import datetime, sys, subprocess, json

from rlberry.agents.torch import PPOAgent #, DQNAgent, MunchausenDQNAgent, A2CAgent
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry.agents.torch.utils.training import model_factory_from_env
from rlberry.wrappers import RescaleRewardWrapper
from rlberry.envs import gym_make

if subprocess.call(("git", "diff-index",
                    "--quiet", "HEAD")):
    print("Repository is dirty, please commit")
    sys.exit(1)

# get the git hash at run time
hash_cmd = ("git", "rev-parse", "HEAD")
revision = subprocess.check_output(hash_cmd)

# parameters definitions
n_fit = 30
params = dict(mlp_size = (256, 256),
              learning_rate = 3e-4,
              gamma = 0.99,
              n_steps = 256,
              seed = 42)

# network architecture
policy_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": params["mlp_size"],  # Network dimensions
    "reshape": False,
    "is_policy": True,
}
value_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": params["mlp_size"],
    "reshape": False,
    "out_size": 1,
}

# environment used
env_ctor, env_kwargs = gym_make, {"id":"Acrobot-v1"}

if __name__ == "__main__":
    # Main agent definition with experiment setting
    manager = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        agent_name="PPOAgent",
        init_kwargs=dict(
            policy_net_fn=model_factory_from_env,
            policy_net_kwargs=policy_configs,
            value_net_fn=model_factory_from_env,
            value_net_kwargs=value_configs,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_steps=params["n_steps"],
        ),
        fit_budget=1e6,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=n_fit,
        parallelization="process",
        mp_context="spawn",
        enable_tensorboard=True,
        seed = params["seed"]
    )
    # do the training
    manager.fit() 
    # we save pre-trained agents and also save the version of rlberry used
    manager.save()

    # Compute the mean and stds over 100 evaluation runs
    # for each of the n_fit agents
    eval_means = []
    eval_stds = []
    
    for id_agent in range(n_fit):
        eval_values = manager.eval_agents(100, agent_id=id_agent)
        eval_means.append(np.mean(eval_values))
        eval_stds.append(np.std(eval_values))

    # export the data.
    results = {
        "data"      : [eval_means, eval_stds],
        "parameters": params,
        "timestamp" : str(datetime.datetime.utcnow()),
        "revision"  : revision,
        "system"    : sys.version}
    
    with open("results-acrobot-ppo.txt", "w") as fd:
        json.dump(results, fd)
