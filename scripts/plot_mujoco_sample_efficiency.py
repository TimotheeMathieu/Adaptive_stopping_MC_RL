import copy
import glob
import os

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

rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["figure.dpi"] = 300
rcParams["font.size"] = 16

rc("text", usetex=False)

ENV_NAMES = {
    "ant": "Ant-v3",
    "halfcheetah": "HalfCheetah-v3",
    "hopper": "Hopper-v3",
    "humanoid": "Humanoid-v3",
    "walker": "Walker2d-v3",
}

XLIMS = {
    "Ant-v3": (1e5, 2e6),
    "HalfCheetah-v3": (1e5, 1e6),
    "Hopper-v3": (1e5, 1e6),
    "Humanoid-v3": (1e5, 2e6),
    "Walker2d-v3": (1e5, 1e6),
}

YLIMS = {
    "Ant-v3": (-1_000, 6_000),
    "HalfCheetah-v3": (-500, 11_500),
    "Hopper-v3": (-500, 3_500),
    "Humanoid-v3": (0, 6_000),
    "Walker2d-v3": (0, 5_000),
}


def set_axes(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_ylabel(ylabel, labelpad=14)


def decorate_axis(ax, wrect=10, hrect=10, ticklabelsize="large"):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))


def load(path):
    timesteps = None
    res_path = os.path.join(os.path.dirname(path), "results/")
    evaluations = {}
    for algo_dir in glob.glob(os.path.join(res_path, "*/")):
        print(algo_dir)
        algo = os.path.basename(os.path.normpath(algo_dir))
        evaluations[algo] = []

        for seed_dir in glob.glob(os.path.join(algo_dir, "*/")):
            seed = os.path.basename(os.path.normpath(seed_dir))

            arr = np.load(os.path.join(seed_dir, "evaluations.npz"), allow_pickle=True)
            ts, evals = arr["timesteps"], arr["evaluations"]

            if timesteps is None:
                timesteps = ts
            else:
                assert np.all(timesteps == ts)
            assert len(timesteps) == len(evals)

            evaluations[algo].append(evals)

        evaluations[algo] = np.vstack(evaluations[algo])
        evaluations[algo] = np.expand_dims(evaluations[algo], axis=1)

    return timesteps, evaluations


def compute_means(evaluations):
    means = lambda y: np.array(
        [metrics.aggregate_mean(y[..., i]) for i in range(y.shape[-1])]
    )
    return rly.get_interval_estimates(evaluations, means)


def plot_sample_efficiency(ax, path):
    env = os.path.basename(os.path.dirname(path))
    env = ENV_NAMES[env]

    algo_dirs = glob.glob(os.path.join(path, "agents/*.sh"))
    algos = [os.path.basename(algo_dir).replace(".sh", "") for algo_dir in algo_dirs]

    mean_path = os.path.join(os.path.dirname(path), "aggregate_evals.csv")
    if os.path.exists(mean_path):
        df = pd.read_csv(mean_path, index_col=0)
        timesteps = np.array(df.index.values)
        means, mean_cis = {}, {}
        for algo in algos:
            means[algo] = np.array(df[algo].values)
            lower = np.array(df[algo + "_lower"].values)
            upper = np.array(df[algo + "_upper"].values)
            mean_cis[algo] = np.stack([lower, upper], axis=0)
    else:
        timesteps, evaluations = load(path)
        means, mean_cis = compute_means(evaluations)
        aux = copy.copy(means)
        for algo in means:
            aux[algo + "_lower"] = mean_cis[algo][0]
            aux[algo + "_upper"] = mean_cis[algo][1]
        df = pd.DataFrame(aux, index=timesteps)
        df.to_csv(mean_path)

    algos = [a.upper() for a in means.keys()]
    means = {a.upper(): means[a] for a in means}
    mean_cis = {a.upper(): mean_cis[a] for a in mean_cis}
    plot_utils.plot_sample_efficiency_curve(
        timesteps,
        means,
        mean_cis,
        algorithms=algos,
        ax=ax,
        xlabel="Time Steps",
        ylabel="Mean of Evaluation Returns",
        color_palette=sns.color_palette("colorblind"),
        labelsize="large",
        ticklabelsize="large",
        marker=" ",
    )

    set_axes(ax, XLIMS[env], YLIMS[env], "Time Steps", "Mean of Evaluation Returns")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_title(env, fontsize="large")
    decorate_axis(ax)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # Sample Efficiency Curves
    # fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    # axs = axs.flatten()

    fig = plt.figure(figsize=(21, 10))
    spec = mpl.gridspec.GridSpec(ncols=6, nrows=2) # 6 columns evenly divides both 2 & 3

    axs = []
    axs.append(fig.add_subplot(spec[0,0:2]))
    axs.append(fig.add_subplot(spec[0,2:4]))
    axs.append(fig.add_subplot(spec[0,4:]))
    axs.append(fig.add_subplot(spec[1,1:3]))
    axs.append(fig.add_subplot(spec[1,3:5]))

    envs = ["ant", "halfcheetah", "hopper", "humanoid", "walker"]
    for i, env in enumerate(envs):
        env_path = os.path.join(args.path, env + "/")
        plot_sample_efficiency(axs[i], env_path)

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=16,
        bbox_to_anchor=(0.5, 1.3),
        bbox_transform=axs[1].transAxes,
        loc="upper center",
        frameon=True,
        fancybox=True,
        ncol=6,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(args.path), "learning_curves.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
