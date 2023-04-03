
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

cmap = mpl.cm.get_cmap("Spectral")
fig = plt.figure(figsize=(5, 2.5))

ax = None
n=0
scale = 1/10
scale2 = 2.4

for case in [1,2]:

    ax = plt.subplot(1, 2, case, frameon=False)

    for i in range(4):
        if case == 1:
            X = np.linspace(-1, 1, 500)
            Y = (norm.pdf(X-i*scale, scale=0.1)+norm.pdf(X+i*scale, scale=0.1))/2
        else:
            X = np.linspace(-1, 2, 500)
            Y = (norm.pdf(X, scale=0.1)+norm.pdf(X-2*i*scale, scale=0.1))/2

        color = cmap(i / 10)
        ax.fill_between(X,  Y + i*scale2, i*scale2, color=color, zorder=100 - i)
        ax.plot(X, Y + i*scale2, color="k", linewidth=0.75, zorder=100 - i)


    ax.yaxis.set_tick_params(tick1On=False)
    if case == 1:
        ax.set_xlim(-0.6, 0.6)
    else:
        ax.set_xlim(-0.3, 1.1)

    ax.axvline(0.0, ls="--", lw=0.75, color="black", zorder=250)


    ax.text(
        -0.1,
        1,
        "Case %d" % (case),
        ha="left",
        va="top",
        weight="bold",
        transform=ax.transAxes,
    )
    deltas = 2*np.arange(4)*scale
    if case == 1:
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_yticks(np.arange(4)*scale2)
        ax.set_yticklabels(["$\Delta=$ {:.1f}".format(np.round(deltas[i-1], 2))  for i in range(1, 5)])
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
            tick.label.set_verticalalignment("bottom")
    else:
        ax.yaxis.set_tick_params(labelleft=False)

    


plt.tight_layout()
plt.savefig("gaussian_mixture_case_1_2.pdf")
#plt.show()
