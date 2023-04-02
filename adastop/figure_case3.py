
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

cmap = mpl.cm.get_cmap("Spectral")
#fig = plt.figure(figsize=(4, 4.5))


n = 10   # number of bars


# Plot
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(5,3))
fig.subplots_adjust(left=0.4, top=1)

ax = None
n=0

ax = plt.subplot(1, 1, 1, frameon=False)

from scipy import stats 

def mg(x, mu1,sigma1, mu2, sigma2):
    return (stats.norm(mu1,sigma1).pdf(x)+stats.norm(mu2,sigma2).pdf(x) )/2
def mt(x, mu1,nu1, mu2, nu2):
    return (stats.t(loc=mu1,df=nu1).pdf(x)+stats.t(loc=mu2,df=nu2).pdf(x) )/2

laws = [lambda x : stats.norm(0,0.1).pdf(x),
        lambda x : stats.norm(0,0.1).pdf(x),
        lambda x : mg(x, -1, 0.1, 1, 0.1),
        lambda x : mg(x, -1, 0.1, 1, 0.1),
        lambda x : mg(x, -0.2, 0.1, 0.2, 0.1),
        lambda x : stats.t(loc=0, df=3).pdf(x),
        lambda x : mg(x, 0, 0.1, 8, 0.1),
        lambda x : mg(x, 0, 0.1, 8, 0.1),
        lambda x : mt(x, 0, 3, 8, 3),
        lambda x : stats.t(loc=8, df = 3).pdf(x),
        ]

scale = 1.4

for i in range(10):
    X = np.linspace(-2, 10, 500)
    Y = laws[-(i+1)](X)

    color = cmap(i / 10)
    ax.fill_between(X,  Y + i*scale, i*scale, color=color, zorder=100 - i)
    ax.plot(X, Y + i*scale, color="k", linewidth=1, zorder=100 - i)


ax.yaxis.set_tick_params(tick1On=False)
ax.set_xlim(-2, 10)

#ax.axvline(0.0, ls="--", lw=0.75, color="black", zorder=250)

names = ["N", "*N", "MG1", "*MG1", "MG2", "tS1", "MG3", "*MG3", "MtS", "tS2"]

ax.yaxis.set_tick_params(labelleft=False)
# ax.yaxis.set_tick_params(labelleft=True)
# ax.set_yticks(np.arange(10)*scale)
# ax.set_yticklabels([names[-(i+1)] for i in range(10)])
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(10)
#     tick.label.set_verticalalignment("bottom")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
latex_names = ["$\mathcal{N}(0, 0.01)$",
               "$\mathcal{N}(0, 0.01)$",
               "$\mathcal{M}_{1/2}^{\mathcal{N}}(-1, 0.01; 1, 0.01)$",
               "$\mathcal{M}_{1/2}^{\mathcal{N}}(-1, 0.01; 1, 0.01)$",
               "$\mathcal{M}_{1/2}^{\mathcal{N}}(-0.2, 0.01; 0.2, 0.01)$",
               "$t(0, 3)$",
               "$\mathcal{M}_{1/2}^{\mathcal{N}}(-1, 0.01; 8, 0.01)$",
               "$\mathcal{M}_{1/2}^{\mathcal{N}}(-1, 0.01; 8, 0.01)$",
               "$\mathcal{M}_{1/2}^{t}(0, 3; 0, 8)$",
               "$t(8,3)$"
               ]


celltxt = [[l] for l in latex_names]
    
the_table = plt.table(cellText=celltxt,
                      rowLabels=[n+" " for n in names],
                      cellLoc='center',
                      loc='left',
                      bbox=(-0.6, 0.045, 0.6, 0.7678),
                      edges="horizontal")
idx = 0

for i, c in enumerate(the_table.get_celld().values()):
    if i % 10 == 0:
        c.visible_edges = 'open'

the_table.auto_set_font_size(False)
the_table.set_fontsize(9)
fig.canvas.draw()   # need to draw the figure twice

#plt.tight_layout()
plt.savefig("case_3.pdf")
#plt.show()
