import gym
import numpy as np
from typing import Union

Noise = Union[float, list]


class StochasticWrapper(gym.Wrapper):
    """
    A generic wrapper to add stochasticity to a discrete action gym environment.
    A probability of taking an action in A\a_t . A gaussian noise for the continuous
    observation, a gaussian noise for the reward scalar.
    """

    def __init__(
        self,
        env: gym.Env,
        prob_wrg_act: float = 0,
        noise_obs: Noise = 0,
        noise_rew: Noise = 0,
    ):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete) and isinstance(
            env.observation_space, gym.spaces.Box
        ), "Need discrete actions, continuous states."
        if isinstance(noise_obs, list):
            assert (
                len(noise_obs) == env.observation_space.shape[0]
            ), "Mu vector of gaussian noise for observation has wrong dimension."

        assert (
            prob_wrg_act >= 0 and prob_wrg_act <= 1
        ), "Value for proba of taking a wrong action is not in [0:1]"

        self.env = env
        self.proba_wrg_act = prob_wrg_act
        self.noise_obs = noise_obs
        self.noise_rew = noise_rew

    def step(self, action):
        act = action
        if np.random.random() < self.proba_wrg_act:
            while act == action:
                act = np.random.randint(self.env.action_space.n)

        next_state, reward, done, info = self.env.step(act)
        next_state += np.random.normal(loc=self.noise_obs, scale=0.1)
        reward += np.random.normal(loc=self.noise_rew, scale=0.1)
        return next_state, reward, done, info


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    env = StochasticWrapper(env, prob_wrg_act = 0.2, noise_obs=0.1, noise_rew = 0)
    env_non_stoch = gym.make("Acrobot-v1")

    s = env.reset()
    ep = 0
    tot = 0
    while True:
        if s[4]<=0: action = 2
        else: action = 0
        s,r,done, _ = env.step(action)
        tot += r
        if done:
            s = env.reset()
            ep += 1
        if ep >= 100:
            break

    print(tot/100)

    s = env_non_stoch.reset()
    ep = 0
    tot = 0
    while True:
        if s[4]<=0: action = 2
        else: action = 0
        s,r,done, _ = env_non_stoch.step(action)
        tot += r
        if done:
            s = env_non_stoch.reset()
            ep += 1
        if ep >= 100:
            break

    print(tot/100)
