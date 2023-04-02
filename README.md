# Adaptive_stopping_MC_RL

Adaptive stopping when evaluating RL agents with evaluation estimated through Monte-Carlo methods.


## Quick start

The main algorithm is in `code/compare_agents.py` and is called `AgentComparator`.

In `code/example_deeprl/example_compare_agent.py` the comparator is used to know whether manager1 and manager2 are deemed different on the environment.

Example of plot describing the process of monitoring an evaluation statistic during the Monte-carlo evaluation, and comparing it to a boundary is the following.
![](boundary.svg)

## Using the Runner

The Runner script `runner.py` is used to compare Deep RL agents using AdaStop. It has options:
```
options:
  -h, --help            show this help message and
                        exit
  --dir DIR, -d DIR     Directory where the comparison
                        will be saved.
  --reset, -r           Reset the comparison.
  --nb-fits NB_FITS, -n NB_FITS
                        Number of fits before each
                        early stopping decision.
  --K K, -K K           Number of checks.
  --B B, -B B           Number of random permutations
                        to approximate the permutation
                        distribution.
  --alpha ALPHA, -a ALPHA
                        Type I error.
  --beta BETA, -b BETA  Early accept parameter.
  --comparisons COMPARISONS [COMPARISONS ...], -c COMPARISONS [COMPARISONS ...]
                        Comparisons to perform.
  --seed SEED, -s SEED  Random seed.
```

And it assumes that it is given the path to a directory `dir` with the following structure:
```
dir
- agents
  - agent1.sh
  - agent2.sh
  ...
```
That is, it assumes that inside `dir` there is another directory called `agents` containing the bash scripts for each agents.
An agent bash script should accept the random seed as it first positional argument, and print the final evaluation to `stdout`.
The directory `experiments/example/` is an example of the directory organization. Let's take a look at `experiments/example/agent1.sh`:
```
#!/bin/bash
echo "Seed: $1"
python -c "import numpy as np;rng = np.random.default_rng($1);print(f'AdaStop Evaluation: {rng.uniform()}')"
```

Throughout the comparison, results and logs are saved to `dir`:
* `dir/adastop_log.txt` contains detailed logs of the comparison.
* `dir/.adastop_comparator.pkl` contains the `MultipleAgentsComparator` instance that is being used in `pickle` format.
* `dir/dirk.csv` contains the evaluations results for step `k`. An example is `example/example1.csv`.

## MuJoCo Experiments

We use four Deep RL algorithms from four different libraries:
1. PPO from [rlberry](https://github.com/rlberry-py/rlberry/).
2. SAC from [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3).
3. DDPG from [CleanRL](https://github.com/vwxyzjn/cleanrl).
4. TRPO from [MushroomRL](https://github.com/MushroomRL/mushroom-rl).

Code can be found on `adastop/deep_rl_agents`. Each library has different requirements, so we recommend the user installs the corresponding requirements in different virtual environments. We adopt the following naming scheme: `as_crl` for CleanRL, `as_rlb` for rlberry, `as_sb` for Stable Baselines 3, and `as_mrl` for MushroomRL. These are the names used in the scripts (found for example in `experiments/ant/agents/ppo.sh`).

In order to rerun a comparison, it suffices to run the following command:
```
$ python runner.py -r -d ./experiments/ant/ -s SEED
```
**WARNING**: Running the command above will delete the previous results (e.g. `ant1.csv`, `ant2.csv`, etc...) in order to rerun the comparison.

## TODO:
- [ ] Have theoretical guarentees for the multiple testin
- [ ] Have speed of convergence for randomization distribution
- [ ] Have some theory on the MC sampling
- [ ] Do comparison with other methods in the litterature
- [ ] Do the simulation studies
- [ ] Do the empirical studies.
- [ ] Possible improvement: adapt Tukey's significant difference for a more performant (and non-pairwise) Multiple testing scheme. Much harder to have theoretical results on this.
## Refs:
- Permutation test https://www.jstor.org/stable/2533442?seq=1
- Stats in RL 1: https://arxiv.org/pdf/1806.08295.pdf
- Stats in RL 2: https://arxiv.org/pdf/1904.06979.pdf
- Nips paper: https://arxiv.org/abs/2108.13264
