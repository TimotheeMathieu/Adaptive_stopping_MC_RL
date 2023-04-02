import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

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

DECISIONS = {
    "smaller": 0,
    "equal": 1,
    "larger": 2,
    "na": 4,
}

ALGORITHMS = ["DDPG", "TRPO", "PPO", "SAC"]

ENV_NAMES = {
    "ant": "Ant-v3",
    "halfcheetah": "HalfCheetah-v3",
    "hopper": "Hopper-v3",
    "humanoid": "Humanoid-v3",
    "walker": "Walker2d-v3",
}

CONFIG = dict(
    n=5, K=6, B=10_000, alpha=0.05, beta=0.0, seed=0
)  # Used in all environments


def get_comparator(results_dir):
    n, K = CONFIG["n"], CONFIG["K"]
    env = os.path.basename(os.path.normpath(results_dir))

    # Seed
    np.random.seed(CONFIG["seed"])

    # Instantiate comparator
    comparator = MultipleAgentsComparator(**CONFIG)

    # Compute partial comparisons
    evals = dict()
    for k in range(K):
        # load data
        path = os.path.join(results_dir, f"{env}{k+1}.csv")
        if not os.path.exists(path):
            break
        df_k = pd.read_csv(path, index_col=0)
        aux = {a: list(df_k[a]) for a in df_k.columns}

        # update partial evaluations
        for a in aux:
            if a not in evals:
                evals[a] = np.array(aux[a])
            else:
                evals[a] = np.concatenate([evals[a], aux[a]])

        # compute partial comparisons
        if k > 1 and "continue" not in comparator.decisions.values():
            break
        comparator.partial_compare(evals)

    return comparator


def plot_comparison(
    axs,
    comparator,
    draw_boxplot=True,
    draw_table=True,
    agent_names=None,
    draw_yticks=True,
    title=None,
):
    # order agents and get evaluations
    if agent_names is None:
        id_sort = np.argsort(comparator.mean_eval_values)
        agent_names = [comparator.agent_names[i] for i in id_sort]
    else:
        id_sort = [comparator.agent_names.index(name) for name in agent_names]
    Z = [comparator.eval_values[name] for name in agent_names]

    # get decisions
    links = np.zeros([len(agent_names), len(agent_names)])
    for i in range(len(comparator.comparisons)):
        c = comparator.comparisons[i]
        decision = comparator.decisions[str(c)]
        if decision == "equal":
            links[c[0], c[1]] = DECISIONS["equal"]
            links[c[1], c[0]] = DECISIONS["equal"]
        elif decision == "larger":
            links[c[0], c[1]] = DECISIONS["larger"]
            links[c[1], c[0]] = DECISIONS["smaller"]
        else:
            links[c[0], c[1]] = DECISIONS["smaller"]
            links[c[1], c[0]] = DECISIONS["larger"]
    links = links[id_sort, :][:, id_sort]
    links = links + DECISIONS["na"] * np.eye(len(links))

    # decision table annotations
    annot = []
    for i in range(len(links)):
        annot_i = []
        for j in range(len(links)):
            if i == j:
                annot_i.append(" ")
            elif links[i, j] == DECISIONS["equal"]:
                annot_i.append("${\\rightarrow  =}\downarrow$")
            elif links[i, j] == DECISIONS["larger"]:
                annot_i.append("${\\rightarrow \geq}\downarrow$")
            else:
                annot_i.append("${\\rightarrow \leq}\downarrow$")
        annot += [annot_i]

    # plot
    if draw_boxplot:
        (ax1, ax2) = axs
    else:
        ax1 = axs

    n_iters = [comparator.n_iters[name] for name in agent_names]
    if draw_table:
        the_table = ax1.table(
            cellText=[n_iters],
            loc="top",
            cellLoc="center",
            rowLabels=["n_iter"],
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)

    # generate a custom colormap
    colors = mpl.colormaps["Pastel1"].colors
    colors = [colors[0], "lightgray", colors[1], "white"]
    cmap = ListedColormap(colors, name="my_cmap")

    # draw the heatmap with the mask and correct aspect ratio
    xlabels = [""] * len(agent_names) if draw_boxplot else agent_names
    ylabels = [""] * len(agent_names) if not draw_yticks else agent_names
    res = sns.heatmap(
        links,
        annot=annot,
        cmap=cmap,
        linewidths=0.5,
        cbar=False,
        xticklabels=xlabels,
        yticklabels=ylabels,
        fmt="",
        ax=ax1,
    )
    pad = 25 if draw_table else 10
    ax1.set_title(title, fontsize="large", pad=pad)

    # drawing the frame
    for _, spine in res.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if draw_boxplot:
        box_plot = ax2.boxplot(Z, labels=np.array(agent_names), showmeans=True)
        for mean in box_plot["means"]:
            mean.set_alpha(0.6)

        ax2.xaxis.set_label([])
        ax2.xaxis.tick_top()
    return {name: comparator.n_iters[name] for name in agent_names}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Directory containing the evaluations."
    )
    parser.add_argument(
        "--draw-table",
        action="store_true",
        help="Whether to draw the `n_iter` table above decision table.",
    )
    parser.add_argument(
        "--draw-boxplot",
        action="store_true",
        help="Whether to draw the boxplot below decision table.",
    )
    args = parser.parse_args()

    envs = ["ant", "halfcheetah", "hopper", "humanoid", "walker"]
    

    n_rows = 4 if args.draw_boxplot else 2
    n_cols = 3
    figsize = (20, 12) if args.draw_boxplot else (12, 7)
    gridspecs = {"height_ratios": [0.75, 1] * 2} if args.draw_boxplot else None

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, gridspec_kw=gridspecs)

    # Decision Tables
    used_budget = {}
    for i, env in enumerate(envs):
        row, col = i // n_cols, i % n_cols

        if args.draw_boxplot:
            ax = (axs[2 * row][col], axs[2 * row + 1][col])
        else:
            ax = axs[row][col]

        results_dir = os.path.join(args.path, f"{env}")
        comparator = get_comparator(results_dir)

        n_iters = plot_comparison(
            ax,
            comparator,
            draw_table=args.draw_table,
            draw_boxplot=args.draw_boxplot,
            agent_names=ALGORITHMS,
            draw_yticks=(col == 0),
            title=ENV_NAMES[env],
        )
        used_budget[env] = n_iters

    # Used Budget
    budget_ax = None
    if args.draw_boxplot:
        gs = axs[2, 2].get_gridspec()
        for ax in axs[2:, -1]:
            ax.remove()
        budget_ax = fig.add_subplot(gs[2:, -1])
    else:
        budget_ax = axs[-1][-1]

    budgets = np.zeros((len(ALGORITHMS), len(envs)))
    for i, algo in enumerate(ALGORITHMS):
        for j, env in enumerate(envs):
            budgets[i, j] = used_budget[env][algo]

    color = sns.color_palette("muted")
    cmap = sns.light_palette(color[3], as_cmap=True)
    sns.heatmap(
        budgets,
        annot=True,
        ax=budget_ax,
        cmap=cmap,
        cbar=False,
        xticklabels=envs,
        yticklabels=[""] * len(ALGORITHMS),
        vmin=0,
        vmax=30,
    )
    env_names = [ENV_NAMES[env][:-3] for env in envs]
    budget_ax.set_xticklabels(env_names, rotation=60, ha="right")
    budget_ax.set_title("Used Budget", fontsize="large", pad=10)

    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(args.path), "comparisons.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
