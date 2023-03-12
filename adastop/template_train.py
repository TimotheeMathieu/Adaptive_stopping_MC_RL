import numpy as np
import json
import os
import sys

from rlberry.agents.torch import DQNAgent
from rlberry.envs import gym_make
from rlberry.manager import AgentManager, evaluate_agents
from rlberry.wrappers import WriterWrapper # In case you want to save anything during training.

import gym
import torch
# * Parameters

hash_cmd = ("git", "rev-parse", "HEAD")
revision = subprocess.check_output(hash_cmd)


parameters = dict(
    env_id="MountainCar-v0",
    layer_sizes=[256, 256],
    q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
    batch_sizes=[1,128],
    max_replay_size=10_000,
    learning_rate=4e-3,
    learning_starts=1000,
    gamma=0.98,
    train_interval=16,
    gradient_steps=8,
    epsilon_init=0.2,
    epsilon_final=0.07,
    epsilon_decay_interval=600,
    fit_budget=1.2e5,
    eval_horizon=500,
    n_fit=1,
    seed=42,
    gpu=False, # Say whether you use GPU!
    gym_version=gym.__version__,
    numpy_version=np.__version__,
    torch_version=torch.__version__,
    adastop_version=revision
)

output_dir_name = "results/DQN_MontainCar/" # experiment folder
os.makedirs(output_dir_name, exist_ok=True)

locals().update(parameters)  # load all the variables defined in parameters dict

model_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": layer_sizes,
    "reshape": False,
}

# * Agent and environment definition
env_ctor = gym_make
env_kwargs = dict(id=env_id)

agents = [AgentManager(
    DQNAgent,
    (env_ctor, env_kwargs),
    init_kwargs=dict(
        q_net_constructor=q_net_constructor,
        q_net_kwargs=model_configs,
        batch_size=batch_size,
        max_replay_size=max_replay_size,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        gamma=gamma,
        train_interval=train_interval,
        gradient_steps=gradient_steps,
        epsilon_init=epsilon_init,
        epsilon_final=epsilon_final,
        epsilon_decay_interval=epsilon_decay_interval,
    ),
    fit_budget=fit_budget,
    eval_kwargs=dict(eval_horizon=eval_horizon),
    n_fit=n_fit,
    parallelization="process",
    mp_context="spawn",
    seed=seed,
    output_dir=output_dir_name+"/My_agent_"+str(batch_size) # one folder for each agent
)
          for batch_size in batch_sizes]


# * Training and saving training data
if __name__ == "__main__":
    # train
    save_paths = []
    for agent in agents:
        agent.fit()
        save_paths.append(str(agent.save()))

    # save parameters
    parameters["save_paths"] = save_paths
    with open(output_dir_name + "parameters.json", "w") as file:
        file.write(json.dumps(parameters))

        
    # optional: check the agent perform as expected
    evaluation = evaluate_agents(agents, n_simulations=16, show=False).values
    assert np.mean(evaluation) > -200


