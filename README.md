This repository can be used to reproduce all the experiments in the paper [AdaStop: sequential testing for efficient and reliable comparisons of Deep RL Agents](https://arxiv.org/abs/2306.10882).

### Structure of the repository

- Folder `toy_examples_xp` contains code for Section 6.1 "Toy examples"
- Folder `Non_adaptive_comparison_xp` contains code for Section 6.2 "Comparison with non-adaptive approach"
- Folder `Deep_RL_xp` contains code for Section 6.3 "AdaStop for Deep Reinforcement Learning"

We provide a `requirements.txt` file which contains the dependencies and their versions at the time of the article.

### What is reproducible?
We aim at having a completely reproducible paper. All the scripts for all the experiments are available and using the makefile all the figures and results of the paper can be reproduced. One can use `make all` (on linux) to do the figures from the article with a reduced size of the experiments so that it runs in a reasonable amount of time, using a virtual environment that is constructed in tmp folder. FOr more courageous people, you can use `make ARG=--full-xp all` which does the exact same figures as the ones in the article.

Remark: due to its complexity, the deep rl experiments from Section 6.3. are not automatically recomputed, we only recompute the adastop algorithm and not the actual training of the agents.

### Using Guix
For (almost) perfect reproducibility, you can use guix with the following commands to fix the versions of the shared libraries (e.g. C libraries)
```
guix time-machine --channels=channels.scm --  shell -CF -m manifest.scm
export LD_LIBRARY_PATH=/lib
make all 
```
