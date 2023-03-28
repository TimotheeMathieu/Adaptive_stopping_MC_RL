import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

scalar_list1 = np.loadtxt("h2g2/evals_sac_hc.txt")
scalar_list2 = np.loadtxt("h2g2/evals_td3_hc.txt")
sns.set_style("whitegrid")
sns.kdeplot(scalar_list1, label = "SAC")
sns.kdeplot(scalar_list2, label  = "TD3")
plt.xlabel("evaluations means")
plt.legend()
plt.savefig("sac_vs_td3_hc.pdf")
