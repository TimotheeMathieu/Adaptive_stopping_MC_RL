# Adaptive_stopping_MC_RL

Adaptive stopping when evaluating RL agents with evaluation estimated through Monte-Carlo methods.

![boundary.svg]

## Quick start

The main algorithm is in `compare_agents.py` and is called `AgentComparator`.

In `example_compare_agent.py` the comparator is used to know whether manager1 and manager2 are deemed different on the environment.



## TODO:
- [ ] See about power and sample-size requirements.
- [ ] See about multiple testing to test multiple agents at the same time. Use [FWE control procedures](https://en.wikipedia.org/wiki/Family-wise_error_rate). Use Bonferroni? Holm's ? Hochberg's ? Or more powerful methods like https://www.jstor.org/stable/27590521#metadata_info_tab_contents (bootstrap) ?
Remarks:
    - There is not really a reason to use Bonferroni as Holm's is proven to be better without any disadvantage.
    - The Bootstrap method may be not conservative enough. In particular with small sample size.
    - Maybe we want to bypass the multiple testing. For instance by testing $H_0: \min_{i \neq j}|\mu_i - \mu_j|=0$ vs $H_1: \forall i\neq j, \, \mu_i \neq \mu_j$.
- [ ] Do parallelization for multiple testing ?
- [ ] Do the simulation studies
- [ ] Do the empirical studies.

## Refs:
- Permutation test https://www.jstor.org/stable/2533442?seq=1
- Stats in RL 1: https://arxiv.org/pdf/1806.08295.pdf
- Stats in RL 2: https://arxiv.org/pdf/1904.06979.pdf
- Nips paper: https://arxiv.org/abs/2108.13264
- Using a subsample of the permutations to speed up computation : https://www.jstor.org/stable/2286069 , https://arxiv.org/pdf/2202.00967.pdf ,


## Linked refs (linked but not exactly the same problem):
- Fast cross validation via sequential testing: https://www.jmlr.org/papers/volume16/krueger15a/krueger15a.pdf
- Sequential Testing for Early Stopping of Online Experiments http://eprints.gla.ac.uk/105812/1/105812.pdf
- A Nonparametric Sequential Test for Online Randomized Experiments https://dl.acm.org/doi/pdf/10.1145/3041021.3054196?casa_token=7IvZtk9bu-IAAAAA:6735VM1feTvIbMxBHYdPP3eV6IPKhd5V8AONvosDopfW9rYLOth2vT7TKqpfHl3vS19pX-1P2nAxWQ
