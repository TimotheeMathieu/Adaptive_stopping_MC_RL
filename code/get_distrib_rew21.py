from gym_stochastic import TwentyOneWithDice
import matplotlib.pyplot as plt
import numpy as np
from ql_sarsa import QLAgent
env = TwentyOneWithDice()
ql = QLAgent(env, epsilon = 0.3)
ql.fit(int(1e6))
# pol_stoch = ql.Q/ql.Q.sum(axis =  1)[:,None]
# print(pol_stoch)
# pol_greedy= ql.Q.argmax(axis=1)[:,None]
rewards = np.zeros(22)
for ep in range(int(1e6)):
    s = env.reset()
    done = False
    tot = 0
    while not done:
        s,r,done,_=env.step(ql.Q[s].argmax())
        tot+=r
    rewards[int(tot)] = rewards[int(tot)]+1
plt.plot(rewards)
plt.show()
