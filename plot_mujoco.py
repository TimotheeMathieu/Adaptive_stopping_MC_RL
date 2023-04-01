import copy
import glob
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

sns.set_style("white")

from matplotlib import rcParams
from matplotlib import rc

rcParams['legend.loc'] = 'best'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['figure.dpi'] = 300
rcParams['font.size'] = 16
#rcParams['font.size'] = 22

rc('text', usetex=False)


ENV_NAMES = {
    'ant': 'Ant-v3',
    'halfcheetah': 'HalfCheetah-v3',
    'hopper': 'Hopper-v3',
    'humanoid': 'Humanoid-v3',
    'walker': 'Walker2d-v3'
}

XLIMS = {
    'Ant-v3': (1e5, 2e6),
    'HalfCheetah-v3': (1e5, 1e6),
    'Hopper-v3': (1e5, 1e6),
    'Humanoid-v3': (1e5, 2e6),
    'Walker2d-v3': (1e5, 1e6),
}

YLIMS = {
    'Ant-v3': (-1_000, 6_000),
    'HalfCheetah-v3': (-500, 11_500),
    'Hopper-v3': (-500, 3_500),
    'Humanoid-v3': (0, 6_000),
    'Walker2d-v3': (0, 5_000),
}


def load(path):
    timesteps = None
    res_path = os.path.join(os.path.dirname(path), 'results/')
    evaluations = {}
    for algo_dir in glob.glob(os.path.join(res_path, '*/')):
        algo = os.path.basename(os.path.normpath(algo_dir))
        evaluations[algo] = []

        for seed_dir in glob.glob(os.path.join(algo_dir, '*/')):
            seed = os.path.basename(os.path.normpath(seed_dir))

            arr = np.load(os.path.join(seed_dir, 'evaluations.npz'), allow_pickle=True)
            ts, evals = arr['timesteps'], arr['evaluations']

            if timesteps is None:
                timesteps = ts
            else:
                assert np.all(timesteps == ts)
            assert len(timesteps) == len(evals)

            evaluations[algo].append(evals)

        evaluations[algo] = np.vstack(evaluations[algo])
        evaluations[algo] = np.expand_dims(evaluations[algo], axis=1)

    return timesteps, evaluations


def fix(path):
    res_path = os.path.join(os.path.dirname(path), 'results/')
    evaluations = {}
    for algo_dir in glob.glob(os.path.join(res_path, '*/')):
        algo = os.path.basename(os.path.normpath(algo_dir))
        evaluations[algo] = []

        for seed_dir in glob.glob(os.path.join(algo_dir, '*/')):
            seed = os.path.basename(os.path.normpath(seed_dir))

            arr = np.load(os.path.join(seed_dir, 'evaluations.npz'), allow_pickle=True)
            ts, evals = arr['timesteps'], arr['evaluations']

            if ts[0] > 0:
                print(ts, evals)
                ts = np.concatenate([[0], ts])
                evals = np.concatenate([[evals[0]], evals])
                print(ts, evals)
            
            if np.all(ts[1: len(ts) // 2 + 1] == ts[len(ts) // 2 + 1:]):
                print(ts, evals)
                half = len(ts) // 2 + 1
                ts = ts[:half]
                evals = evals[:half]
                print(ts, evals)

            np.savez(os.path.join(seed_dir, 'evaluations.npz'), timesteps=ts, evaluations=evals)


def compute_iqms(evaluations):
    #iqm = lambda y: np.array([metrics.aggregate_iqm(y[..., i]) for i in range(y.shape[-1])])
    iqm = lambda y: np.array([metrics.aggregate_mean(y[..., i]) for i in range(y.shape[-1])])
    return rly.get_interval_estimates(evaluations, iqm)


def set_axes(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_ylabel(ylabel, labelpad=14)


def decorate_axis(ax, wrect=10, hrect=10, ticklabelsize='large'):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))


def plot_sample_efficiency(ax, path):
    env = os.path.basename(os.path.dirname(path))
    env = ENV_NAMES[env]

    timesteps, evaluations = load(path)
    
    iqm_path = os.path.join(os.path.dirname(path), 'iqms.csv')
    if os.path.exists(iqm_path):
        df = pd.read_csv(iqm_path, index_col=0)
        iqms, iqm_cis = {}, {}
        for algo in evaluations:
            iqms[algo] = np.array(df[algo].values)
            lower = np.array(df[algo + '_lower'].values)
            upper = np.array(df[algo + '_upper'].values)
            iqm_cis[algo] = np.stack([lower, upper], axis=0)
    else:
        iqms, iqm_cis = compute_iqms(evaluations)
        aux = copy.copy(iqms)
        for algo in iqms:
            aux[algo + '_lower'] = iqm_cis[algo][0]
            aux[algo + '_upper'] = iqm_cis[algo][1]
        df = pd.DataFrame(aux, index=timesteps)
        df.to_csv(iqm_path)

    algos = [a.upper() for a in iqms.keys()]
    iqms = {a.upper(): iqms[a] for a in iqms}
    iqm_cis = {a.upper(): iqm_cis[a] for a in iqm_cis}
    plot_utils.plot_sample_efficiency_curve(
        timesteps, iqms, iqm_cis, algorithms=algos, ax=ax,
            xlabel='Time Steps',
            ylabel='Mean of Evaluation Returns',
            color_palette=sns.color_palette('colorblind'),
            labelsize='large', ticklabelsize='large',
            marker=' ')

    set_axes(ax, XLIMS[env], YLIMS[env], 'Time Steps', 'Mean of Evaluation Returns')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.grid(True, axis='y', alpha=0.5)
    # ax.legend(loc='upper left', fontsize='large')
    ax.set_title(env, fontsize='large')
    decorate_axis(ax)


DECISIONS = {
    'smaller': 0,
    'equal': 1,
    'larger': 2,
    'na': 4,
}

def plot_comparison(axs, path, draw_boxplot=True, draw_n_iter=True, 
    agent_names=None, draw_yticks=True):
    env = os.path.basename(os.path.dirname(path))
    env = ENV_NAMES[env]

    # load comparator
    comparator_path = os.path.join(os.path.dirname(path), 'comparator.pkl')
    with open(comparator_path, 'rb') as f:
        comp = pickle.load(f)

    # order agents and get evaluations
    if agent_names is None:
        id_sort = np.argsort(comp.mean_eval_values)
        agent_names = [comp.agent_names[i] for i in id_sort]
    else:
        id_sort = [comp.agent_names.index(name) for name in agent_names]
    Z = [comp.eval_values[name] for name in agent_names]

    # get decisions
    links = np.zeros([len(agent_names),len(agent_names)])
    for i in range(len(comp.comparisons)):
        c = comp.comparisons[i]
        decision = comp.decisions[str(c)]
        if decision == "equal":
            links[c[0],c[1]] = DECISIONS['equal']
            links[c[1],c[0]] = DECISIONS['equal']
        elif decision == "larger":
            links[c[0],c[1]] = DECISIONS['larger']
            links[c[1],c[0]] = DECISIONS['smaller']
        else:
            links[c[0],c[1]] = DECISIONS['smaller']
            links[c[1],c[0]] = DECISIONS['larger']
    links = links[id_sort,:][:,id_sort]
    links = links + DECISIONS['na'] * np.eye(len(links))

    # decision table annotations
    annot = []
    for i in range(len(links)):
        annot_i = []
        for j in range(len(links)):
            if i == j:
                annot_i.append(" ")                    
            elif links[i,j] == DECISIONS['equal']:
                annot_i.append("${\\rightarrow  =}\downarrow$")
            elif links[i,j] == DECISIONS['larger']:
                annot_i.append("${\\rightarrow \geq}\downarrow$")
            else:
                annot_i.append("${\\rightarrow \leq}\downarrow$")
        annot += [annot_i]
    
    # plot
    if draw_boxplot:
        (ax1, ax2) = axs
    else:
        ax1 = axs

    n_iterations = [comp.n_iters[name] for name in agent_names]
    if draw_n_iter:
        the_table = ax1.table(
            cellText=[n_iterations], loc="top", cellLoc="center", rowLabels=["n_iter"], 
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)

    # generate a custom colormap
    from matplotlib.colors import ListedColormap
    colors = mpl.colormaps['Pastel1'].colors
    colors = [colors[0], 'lightgray', colors[1], 'white']
    cmap = ListedColormap(colors, name="my_cmap")

    # draw the heatmap with the mask and correct aspect ratio
    xlabels = [''] * len(agent_names) if draw_boxplot else agent_names
    ylabels = [''] * len(agent_names) if not draw_yticks else agent_names
    res = sns.heatmap(links, annot=annot, cmap=cmap, linewidths=.5, cbar=False,
        xticklabels=xlabels, yticklabels=ylabels, fmt='', ax=ax1)
    pad = 25 if draw_n_iter else 10
    ax1.set_title(env, fontsize='large', pad=pad)

    # drawing the frame
    for _, spine in res.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if draw_boxplot:
        box_plot = ax2.boxplot(Z, labels=np.array(agent_names), showmeans=True)
        for mean in box_plot['means']:
            mean.set_alpha(0.6)

        ax2.xaxis.set_label([])
        ax2.xaxis.tick_top()
    return {name: comp.n_iters[name] for name in agent_names}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    # Comparisons
    fig, axs = plt.subplots(4, 3, figsize=(20, 12), gridspec_kw={"height_ratios": [0.75, 1, 0.75, 1]})

    n_iter = {}
    envs = ['ant', 'halfcheetah', 'hopper', 'humanoid', 'walker']
    for i, env in enumerate(envs):
        row, col = i // 3, i % 3
        env_path = os.path.join(args.path, env + '/')
        ns = plot_comparison((axs[2 * row][col], axs[2 * row + 1][col]), env_path)
        n_iter[env] = ns

    fig.subplots_adjust(hspace=0.01, wspace=0.01)

    gs = axs[2, 2].get_gridspec()
    for ax in axs[2:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[2:, -1])

    aux = []
    algos = list(n_iter[envs[0]].keys())
    aux = np.zeros((len(envs), len(algos)))
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            aux[i, j] = n_iter[env][algo]

    color = sns.color_palette('muted')
    cmap = sns.light_palette(color[3], as_cmap=True)
    sns.heatmap(aux, annot=True, ax=axbig, cmap=cmap, cbar=False,
        xticklabels=algos, yticklabels=envs, vmin=0, vmax=30)
    axbig.set_title('Used Budget', fontsize='large', pad=10)

    env_names = [ENV_NAMES[env][:-3] for env in envs]
    axbig.set_yticklabels(env_names, rotation=60, ha='right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(args.path), 'comparisons_with_table_boxplot.pdf'),
        format='pdf', bbox_inches='tight')
    plt.close()

    # Comparisons without Table
    fig, axs = plt.subplots(4, 3, figsize=(20, 12), gridspec_kw={"height_ratios": [0.75, 1, 0.75, 1]})

    n_iter = {}
    envs = ['ant', 'halfcheetah', 'hopper', 'humanoid', 'walker']
    for i, env in enumerate(envs):
        row, col = i // 3, i % 3
        env_path = os.path.join(args.path, env + '/')
        ns = plot_comparison((axs[2 * row][col], axs[2 * row + 1][col]), env_path, draw_n_iter=False)
        n_iter[env] = ns

    fig.subplots_adjust(hspace=0.01, wspace=0.01)

    gs = axs[2, 2].get_gridspec()
    for ax in axs[2:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[2:, -1])

    aux = []
    algos = list(n_iter[envs[0]].keys())
    aux = np.zeros((len(envs), len(algos)))
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            aux[i, j] = n_iter[env][algo]

    color = sns.color_palette('muted')
    cmap = sns.light_palette(color[3], as_cmap=True)
    sns.heatmap(aux, annot=True, ax=axbig, cmap=cmap, cbar=False,
        xticklabels=algos, yticklabels=envs, vmin=0, vmax=30)
    axbig.set_title('Used Budget', fontsize='large', pad=10)

    env_names = [ENV_NAMES[env][:-3] for env in envs]
    axbig.set_yticklabels(env_names, rotation=60, ha='right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(args.path), 'comparisons_with_boxplot.pdf'),
        format='pdf', bbox_inches='tight')
    plt.close()

    # Comparisons without Boxplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    n_iter = {}
    envs = ['ant', 'halfcheetah', 'hopper', 'humanoid', 'walker']
    for i, env in enumerate(envs):
        row, col = i // 3, i % 3
        env_path = os.path.join(args.path, env + '/')
        ns = plot_comparison(axs[row][col], env_path, draw_boxplot=False)
        n_iter[env] = ns

    fig.subplots_adjust(hspace=0.01, wspace=0.01)

    axbig = axs[-1][-1]

    aux = []
    algos = list(n_iter[envs[0]].keys())
    aux = np.zeros((len(envs), len(algos)))
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            aux[i, j] = n_iter[env][algo]

    color = sns.color_palette('muted')
    cmap = sns.light_palette(color[3], as_cmap=True)
    sns.heatmap(aux, annot=True, ax=axbig, cmap=cmap, cbar=False,
        xticklabels=algos, yticklabels=envs, vmin=0, vmax=30)
    axbig.set_title('Used Budget', fontsize='large', pad=10)

    env_names = [ENV_NAMES[env][:-3] for env in envs]
    axbig.set_yticklabels(env_names, rotation=60, ha='right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(args.path), 'comparisons_with_table.pdf'),
        format='pdf', bbox_inches='tight')
    plt.close()

    # Comparisons without Table and Boxplot
    algos = ['DDPG', 'TRPO', 'PPO', 'SAC']
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    n_iter = {}
    envs = ['ant', 'halfcheetah', 'hopper', 'humanoid', 'walker']
    for i, env in enumerate(envs):
        row, col = i // 3, i % 3
        env_path = os.path.join(args.path, env + '/')
        ns = plot_comparison(axs[row][col], env_path, draw_boxplot=False, draw_n_iter=False,
            draw_yticks=(col == 0), agent_names=algos)
        n_iter[env] = ns

    fig.subplots_adjust(hspace=0.01, wspace=0.01)

    axbig = axs[-1][-1]

    aux = np.zeros((len(algos), len(envs)))
    for i, algo in enumerate(algos):
        for j, env in enumerate(envs):
            aux[i, j] = n_iter[env][algo]

    color = sns.color_palette('muted')
    cmap = sns.light_palette(color[3], as_cmap=True)
    sns.heatmap(aux, annot=True, ax=axbig, cmap=cmap, cbar=False,
        xticklabels=envs, yticklabels=['']*len(algos), vmin=0, vmax=30)
    axbig.set_title('Used Budget', fontsize='large', pad=10)

    env_names = [ENV_NAMES[env][:-3] for env in envs]
    axbig.set_xticklabels(env_names, rotation=60, ha='right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(args.path), 'comparisons.pdf'),
        format='pdf', bbox_inches='tight')
    plt.close()

    # Sample Efficiency Curves
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    axs = axs.flatten()

    envs = ['ant', 'halfcheetah', 'hopper', 'humanoid', 'walker']
    for i, env in enumerate(envs):
        env_path = os.path.join(args.path, env + '/')
        plot_sample_efficiency(axs[i], env_path)

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=16,
        bbox_to_anchor=(0.5, 1.3), bbox_transform=axs[1].transAxes,
        loc='upper center', frameon=True, fancybox=True, ncol=6)
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(args.path), 'learning_curves.pdf'),
        format='pdf', bbox_inches='tight')
    plt.close()


