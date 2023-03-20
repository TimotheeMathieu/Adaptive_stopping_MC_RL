import numpy as np
import matplotlib.pyplot as plt

#
# for n in range(2, 6):
#     for k in range(1, 16):
#         decisions = np.load("h2g2/decision_n{}_k{}.npy".format(n,k))
#         comparisons = np.load("h2g2/comparison_n{}_k{}.npy".format(n,k))
#         hists = []
#         for i, deci in enumerate(decisions):
#             if deci:
#                 hists.append(comparisons[i])
#         plt.hist(hists)
#         plt.show()

decisions = np.load("h2g2/decision_n{}_k{}.npy".format(5,5))
comparisons = np.load("h2g2/comparison_n{}_k{}.npy".format(5,5))
hists = []
for i, deci in enumerate(decisions):
    if deci:
        hists.append(comparisons[i])
plt.hist(hists)
plt.show()



l = np.zeros(25)



for i, compar in enumerate(comparisons):
    l[int(compar)-1] += 1

plt.bar(np.arange(1,26), l)
plt.show()
