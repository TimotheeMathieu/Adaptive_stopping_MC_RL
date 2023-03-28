
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

cmap = mpl.cm.get_cmap("Spectral")
fig = plt.figure(figsize=(4, 4))

ax = None
n=0
ax = plt.subplot(1, 1, 1, frameon=False, sharex=ax)
for i in range(5):
    X = np.linspace(-8, 8, 500)
    Y = (norm.pdf(X-i)+norm.pdf(X+i))/2

    ax.plot(X, Y + i/2, color="k", linewidth=0.75, zorder=100 - i)
    color = cmap(i / 50)
    ax.fill_between(X,  Y + i/2, i/2, color=color, zorder=100 - i)


ax.yaxis.set_tick_params(tick1On=False)
ax.set_xlim(-8, 8)
ax.axvline(0.0, ls="--", lw=0.75, color="black", zorder=250)


ax.yaxis.set_tick_params(labelleft=True)
ax.set_yticks(np.arange(5)/2)
ax.set_yticklabels(["$\Delta=$ %d" % i for i in range(1, 6)])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
    tick.label.set_verticalalignment("bottom")



plt.tight_layout()
#plt.savefig("gaussians.pdf")
plt.show()
