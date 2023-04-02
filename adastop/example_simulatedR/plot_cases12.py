import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np


import matplotlib
import seaborn as sns
home_folder = os.environ["HOME"]
workdir = os.path.join(home_folder,"Adaptive_stopping_MC_RL","adastop", "example_simulatedR")


from adastop import MultipleAgentsComparator


# to read csv for old version of adastop
def powers_case2(fname1, fname2):

    decs_df = pd.read_csv(fname1, index_col = 0)
    n_iter_df  = pd.read_csv(fname2, index_col = 0)


    power = {}
    power_std = {}
    power_confidence_interval = {}
    n_iter_avg = {}
    sqrt_n = np.sqrt(len(decs_df))
    print(sqrt_n)

    for dmu in decs_df.columns:
        p = np.array(decs_df[dmu]) == "reject"
        power[dmu] = np.mean(p)
        power_std[dmu] = np.std(p)
        power_confidence_interval[dmu] = np.std(p)/sqrt_n*3
    for k in n_iter_df.columns:
        n_iter_avg[k] = np.mean(n_iter_df[k])

    powers, power_stds, power_confidence_intervals = dict(sorted(power.items())),  dict(sorted(power_std.items())), dict(sorted(power_confidence_interval.items())) 

    return powers, power_stds, power_confidence_intervals

# to read csv for new version of adastop
def powers_case1(fname1, fname2):

    decs_df = pd.read_csv(fname1, index_col = 0)
    n_iter_df  = pd.read_csv(fname2, index_col = 0)

    power = {}
    power_std = {}
    power_confidence_interval = {}
    n_iter_avg = {}
    sqrt_n = np.sqrt(len(decs_df))
    print(sqrt_n)

    for dmu in decs_df.columns:
        p = np.array(decs_df[dmu]) != "equal"
        power[dmu] = np.mean(p)
        power_std[dmu] = np.std(p)
        power_confidence_interval[dmu] = np.std(p)/sqrt_n*3
    for k in n_iter_df.columns:
        n_iter_avg[k] = np.mean(n_iter_df[k])

    powers, power_stds, power_confidence_intervals = dict(sorted(power.items())),  dict(sorted(power_std.items())), dict(sorted(power_confidence_interval.items())) 
    return powers, power_stds, power_confidence_intervals


def create_data_and_annot(p, ci, labels = None):

    power_df = pd.DataFrame(p).T
    error_df = pd.DataFrame(ci).T

    annot = {}

    for i in power_df.index:
        annot[i] = {}
        for k in power_df.columns:
            annot[i][k] = ("{:.3f}\n($\\pm${:.3f})".format(power_df.loc[i,k], error_df.loc[i,k]))
    annot_df = pd.DataFrame(annot).T

    if labels is not None:   
        annot_df.columns = labels
        power_df.columns = labels

    return power_df, annot_df


def plot_heatmap(data, annot, fname):

    plt.rcParams.update({'font.size': 15})


    fig, ax1 = plt.subplots(1,1, figsize=(20,3))


    sns.heatmap(data, annot=annot, fmt="s", linewidths=.5, ax=ax1, cmap = "coolwarm", norm = matplotlib.colors.PowerNorm(0.18), cbar_kws={"ticks": [0, 0.05, 0.2, 0.4, 0.6, 0.8, 1]})
    cbar = ax1.collections[0].colorbar

    ticklabs = cbar.ax.get_yticklabels()
    # print(ticklabs)

    cbar.ax.set_yticklabels(ticklabs, fontsize=11)

    plt.xlabel("$\Delta$")
    plt.ylabel("rejection frequency")


    plt.savefig(fname, bbox_inches='tight')




if __name__ == "__main__":

    labels = [r"0", r"$\frac{1}{9}$", r"$\frac{2}{9}$", r"$\frac{3}{9}$", r"$\frac{4}{9}$", r"$\frac{5}{9}$", r"$\frac{6}{9}$",r"$\frac{7}{9}$", r"$\frac{8}{9}$", r"1"]


    fname = os.path.join(workdir, "Case12_plot.pdf")

    # #for the plots from paper
    # powers, power_stds, power_confidence_intervals = powers_case2("exp2_5000_decs.csv", "exp2_5000_niter.csv")
    # powers2, power_stds2, power_confidence_intervals2 = powers_case1("exp1_5000_decs.csv", "exp1_5000_niter.csv")

    powers2, power_stds2, power_confidence_intervals2 = powers_case1("exp1_10_decs.csv", "exp1_10_niter.csv")
    powers, power_stds, power_confidence_intervals = powers_case1("exp2_10_decs.csv", "exp2_10_niter.csv")


    p = {"Case 2": powers, "Case 1": powers2,}
    ci = {"Case 2": power_confidence_intervals, "Case 1": power_confidence_intervals2,}

    data, annot = create_data_and_annot(p, ci, labels)
    plot_heatmap(data, annot, fname)

    



