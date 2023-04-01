import copy
import glob
import os

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

from adastop import MultipleAgentsComparator

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

    timesteps, evaluations = load(path)

    iqm_path = os.path.join(os.path.dirname(path), "iqms.csv")
    if os.path.exists(iqm_path):
        df = pd.read_csv(iqm_path, index_col=0)
        iqms, iqm_cis = {}, {}
        for algo in evaluations:
            iqms[algo] = np.array(df[algo].values)
            lower = np.array(df[algo + "_lower"].values)
            upper = np.array(df[algo + "_upper"].values)
            iqm_cis[algo] = np.stack([lower, upper], axis=0)
    else:
        iqms, iqm_cis = compute_means(evaluations)
        aux = copy.copy(iqms)
        for algo in iqms:
            aux[algo + "_lower"] = iqm_cis[algo][0]
            aux[algo + "_upper"] = iqm_cis[algo][1]
        df = pd.DataFrame(aux, index=timesteps)
        df.to_csv(iqm_path)

    algos = [a.upper() for a in iqms.keys()]
    iqms = {a.upper(): iqms[a] for a in iqms}
    iqm_cis = {a.upper(): iqm_cis[a] for a in iqm_cis}
    plot_utils.plot_sample_efficiency_curve(
        timesteps,
        iqms,
        iqm_cis,
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
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    axs = axs.flatten()

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