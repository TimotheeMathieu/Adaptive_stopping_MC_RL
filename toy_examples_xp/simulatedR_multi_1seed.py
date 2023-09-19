import os

from utils import *
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from rlberry.utils.logging import set_level
set_level('ERROR')

workdir = os.path.realpath(os.path.dirname(__file__))


def plot_results(comp, fname, agent_names = None, axes= None):

    plt.rcParams.update({'font.size': 15})

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={'width_ratios':  [6,7]})
    fig.tight_layout()


    id_sort = np.argsort(comp.mean_eval_values)

    if agent_names is None:
        agent_names = comp.agent_names

    links = np.zeros([len(agent_names),len(agent_names)])


    decisions = {
            'smaller': 2,
            'equal': 0,
            'larger': 1,
            'na': 4,
        }


    for i in range(len(comp.comparisons)):
        c = comp.comparisons[i]
        decision = comp.decisions[str(c)]
        if decision == "equal":
            links[c[0],c[1]] = decisions['equal']
            links[c[1],c[0]] = decisions['equal']
        elif decision == "larger":
            links[c[0],c[1]] = decisions['larger']
            links[c[1],c[0]] = decisions['smaller']
        else:
            links[c[0],c[1]] = decisions['smaller']
            links[c[1],c[0]] = decisions['larger']


    links = links[id_sort,:][:, id_sort]
    links = links + decisions['na'] * np.eye(len(links))
    print(links)
    
    annot = []
    for i in range(len(links)):
        annot_i = []
        for j in range(len(links)):
            if i == j:
                annot_i.append(" ")                    
            elif links[i,j] == 0:
                annot_i.append("${\\rightarrow  =}\downarrow$")
            elif links[i,j] == 1:
                annot_i.append("${\\rightarrow \geq}\downarrow$")
            else:
                annot_i.append("${\\rightarrow  \leq}\downarrow$")
        annot+= [annot_i]

    n_iterations = [comp.n_iters[comp.agent_names[i]] for i in id_sort]
    the_table = ax2.table(
        cellText=[n_iterations], rowLabels=["n_iter"], loc="top", cellLoc="center", edges="open"
    )

    from matplotlib.colors import ListedColormap

    colors = matplotlib.colormaps['Pastel1'].colors
    colors = ['lightgray',colors[1], colors[0], 'white']
    cmap = ListedColormap(colors, name="my_cmap")
    annot = np.array(annot)
    
    res = sns.heatmap(links, annot = annot, cmap=cmap, #vmax=2, center=0,
                      linewidths=.5, ax =ax2, 
                      cbar=False, yticklabels=np.array(agent_names)[id_sort],  
                      xticklabels=np.array(agent_names)[id_sort],fmt='')


    # Drawing the frame
    for _, spine in res.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)


    Z = [comp.eval_values[comp.agent_names[i]] for i  in id_sort]
    # Set up the matplotlib figure


    pos = np.arange(len(Z))



    sns.set_style("whitegrid")
    sns.violinplot(data=Z, palette="Pastel1", bw = .2, linewidth=1.5, scale="width", ax = ax3)

    ax3.xaxis.set_label([])
    ax3.set_xticks(pos)
    ax3.set_xticklabels(np.array(agent_names)[id_sort])

    plt.savefig(fname, bbox_inches='tight')



labels = ["N", "*N", "*MG1", "MG1", "MG2", "tS1", "MG3", "*MG3", "MtS", "tS2"]

def exp4():
    managers = [create_agents("single", labels[0], type = "normal", drift =0),
                create_agents("single", labels[1], type = "normal", drift =0),
                create_agents("mixture", labels[2], mus = [1.,-1.], probas=[0.5,0.5]),
                create_agents("mixture", labels[3], mus = [1.,-1.], probas=[0.5,0.5]),
                create_agents("mixture", labels[4], mus = [0.2,-0.2], probas=[0.5,0.5]),
                create_agents("single", labels[5], type = "student", df = 3, drift = 0.),#
                create_agents("mixture", labels[6], mus = [0.,8.], probas=[0.5,0.5]),
                create_agents("mixture", labels[7], mus = [0.,8.], probas=[0.5,0.5]),
                create_agents("mixture", labels[8], type = "student", mus = [0. ,8.], probas = [0.5,0.5], df = 3),#
                create_agents("single", labels[9], type = "student", drift = 8., df=3),
                ]
    
    return managers


if __name__ == "__main__":

    res = []
    restime = []
    p_vals = []

    EXP = "exp4"
    alpha = 0.05
    M = 1

    seeds = np.arange(M)
    K_list = np.array([5])
    n_list = np.array([5])
    B_list = np.array([10**4])

    mesh = np.meshgrid(seeds, K_list, n_list, B_list)

    seed_iter = mesh[0].reshape(-1)
    K_iter = mesh[1].reshape(-1)
    n_iter = mesh[2].reshape(-1)
    B_iter = mesh[3].reshape(-1)

    num_comb = len(mesh[0].reshape(-1))

    def decision(**kwargs):
        exp_name = kwargs["exp_name"]
        os.makedirs(os.path.join(workdir, "example_simulatedR", "multc_sd_res", exp_name), exist_ok=True)
        filename = os.path.join(workdir, "example_simulatedR", "multc_sd_res", exp_name, "result_K={}-n={}-B={}-seed={}.pickle".format(kwargs["K"], kwargs["n"], kwargs["B"], kwargs["seed"]))
        comparator = RlberryComparator(n = kwargs["n"], K = kwargs["K"], B = kwargs["B"], alpha = alpha, seed=kwargs["seed"], beta=0.01, n_evaluations=1)
        if exp_name == "exp4":
            managers = exp4()
        else:
            raise ValueError
        comparator.compare(managers, verbose = False)
        with open(filename, "wb") as f:
            pickle.dump([kwargs, comparator], f)
        return comparator

    def decision_par(i):
        return decision(seed = seed_iter[i], n = n_iter[i], K = K_iter[i], B = B_iter[i], exp_name = EXP)

    res = Parallel(n_jobs=-5, backend="multiprocessing")(delayed(decision_par)(i) for i in tqdm(range(num_comb))) # n_jobs=1 for debugging


    path = os.path.join("results", "toy_example_multiagent_test_plot.pdf")

    plot_results(res[0], path, labels)


    # print(comparator.decisions,
    #   comparator.comparisons,
    #   comparator.rejected_decision)
