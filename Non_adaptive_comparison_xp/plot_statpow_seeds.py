import matplotlib.pyplot as plt
import json


def get_x_y_from_json():
    x , y = [], []
    f = open("./data/res_true_positives.json")
    dict_ = json.load(f)
    # k =5
    for k in range(1, 16):
        x.append(dict_["n=5, k={}".format(k)]["true_positives"])
        y.append(dict_["n=5, k={}".format(k)]["effective_comparisons"])
    return x, y


x_adastop, y_adastop = get_x_y_from_json()
# Data from Colas et. al. 2019
x_ttest = [0, 0, 0.379, 0.571, 0.767, 0.827, 0.981, 0.999, 1]
x_welch = [0, 0.411, 0.475, 0.629, 0.793, 0.933, 0.983, 0.999, 1]
x_mw = [0.113, 0.196, 0.388, 0.522, 0.664, 0.780, 0.837, 0.892, 0.95]
x_rttest = [0.059, 0.125, 0.304, 0.482, 0.638, 0.778, 0.842, 0.891, 0.951]
x_boot = [0, 0.5, 0.57, 0.632, 0.705, 0.809, 0.862, 0.93, 0.966]
x_perm = [0, 0.380, 0.475, 0.575, 0.670, 0.782, 0.835, 0.907, 0.953]
y_non_adapt = [2, 3, 5, 7, 10, 15, 20, 30, 40]
with_5more = [7,8,10,12, 15,20, 25, 35, 45]
plt.plot(x_ttest, with_5more, label = "t-test")
plt.plot(x_welch, with_5more, label = "Welch")
plt.plot(x_mw, with_5more, label = "Mann-Whit.")
plt.plot(x_boot, with_5more, label = "bootstrap.")
plt.plot(x_perm, with_5more, label = "permut.")
plt.plot(x_adastop, y_adastop, label = "AdaStop n=5", linewidth=4)
plt.grid()
plt.legend()
plt.ylabel("Nb of Random Seeds")
plt.xlabel("Statistical Power")
plt.vlines([0.8], [5], [45], linestyles="dashed", color = "black")
plt.savefig("power-seeds-nonadapt.pdf")