from rlberry.agents.torch import A2CAgent, PPOAgent
from multiple_managers import MultipleManagers
from rlberry.manager import AgentManager
from rlberry.envs import gym_make
import numpy as np
from rlberry.agents.torch.utils.training import model_factory_from_env

# set_level('INFO')

# Using parameters from deeprl quick start
policy_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": (64, 64),  # Network dimensions
    "reshape": False,
    "is_policy": True,
}

critic_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (64, 64),
    "reshape": False,
    "out_size": 1,
}
n_steps = 1e5
batch_size = 128

dirnames = ["rlberry_data/A2C", "rlberry_data/A2Ct", "rlberry_data/PPO"]
if __name__ == "__main__":
    agents = []
    agents.append(
        AgentManager(
            A2CAgent,
            (gym_make, dict(id="CartPole-v1")),
            agent_name="A2CAgent",
            fit_budget=n_steps,
            eval_kwargs=dict(eval_horizon=500),
            n_fit=5,
            parallelization="process",
            mp_context="spawn",
            default_writer_kwargs={"log_interval": 5},
            output_dir=dirnames[0],
        )
    )

    agents.append(
        AgentManager(
            A2CAgent,
            (gym_make, dict(id="CartPole-v1")),
            agent_name="A2CAgent_tuned",
            init_kwargs=dict(
                policy_net_fn=model_factory_from_env,
                policy_net_kwargs=policy_configs,
                value_net_fn=model_factory_from_env,
                value_net_kwargs=critic_configs,
                entr_coef=0.0,
                batch_size=1024,
                optimizer_type="ADAM",
                learning_rate=1e-3,
            ),
            fit_budget=n_steps,
            eval_kwargs=dict(eval_horizon=500),
            n_fit=5,
            parallelization="process",
            mp_context="spawn",
            default_writer_kwargs={"log_interval": 5},
            output_dir=dirnames[1],
        )
    )
    agents.append(
        AgentManager(
            PPOAgent,
            (gym_make, dict(id="CartPole-v1")),
            agent_name="PPOAgent",
            fit_budget=n_steps,
            eval_kwargs=dict(eval_horizon=500),
            n_fit=5,
            parallelization="process",
            mp_context="spawn",
            default_writer_kwargs={"log_interval": 5},
            output_dir=dirnames[2],
        )
    )

    multimanager = MultipleManagers()

    for agent in agents:
        multimanager.append(agent)

    multimanager.run()
    # multimanager.load(dirnames)
    multimanager.pilot_study()
    # This last command recommand a number of seeds.
    # Change the number of seed to the one advised and run the manager again befor doing stat_test
    # multimanager.stat_test()
