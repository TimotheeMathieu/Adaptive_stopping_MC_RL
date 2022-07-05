from alt_agent_manager import AgentManager
from rlberry.agents.torch import A2CAgent
from rlberry.envs import gym_make
from compare_agents import AgentComparator
import time
# GST definition

K = 7  # at most 5 groups
alpha = 0.05
n = 2 # size of a group

comparator = AgentComparator(n, K, alpha)

# DeepRL agent definition
env_ctor = gym_make
env_kwargs = dict(id="CartPole-v1")
seed = 42
budget = 1e2


if __name__ == "__main__":
    
    manager1 = (
        A2CAgent,
        dict(train_env=(env_ctor, env_kwargs),
        fit_budget=budget,
        seed=seed,
        eval_kwargs=dict(eval_horizon=500),
        init_kwargs=dict(
            learning_rate=1e-3, entr_coef=0.0 
        ),
        parallelization="process",
        mp_context="forkserver",)
    )
    manager2 = (
        A2CAgent,
        dict(train_env=(env_ctor, env_kwargs),
        fit_budget=budget,
        seed=seed,
        init_kwargs=dict(
            learning_rate=1e-3,  
            entr_coef=0.0,
            #batch_size=1024,
        ),
        eval_kwargs=dict(eval_horizon=500),
        agent_name="A2C_tuned",
        parallelization="process",
        mp_context="forkserver",)
    )
    M = 1
    res = []
    for _ in range(M):
        a = time.time()
        comparator.compare(manager2, manager1)
        res.append(comparator.decision)
        print("Time: ",time.time()-a)
    
