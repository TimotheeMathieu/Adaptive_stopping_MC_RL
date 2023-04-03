This folder contains the code used to reproduce Section 6.2 "Comparison with non-adaptive approach".

In this folder, we provide the script `exp_true_positives.py` used to reproduce Table 1.
The data from the paper "A hitchhiker’s guide to statistical comparisonsof reinforcement learning algorithm" by Colas, C., Sigaud, O., Oudeyer, P.Y. is included in the folder `data` under the names `evals_sac_hc.txt` and `evals_td3_hc.txt`.

In the script, we changed the parameters of the problem to smaller scale experiment in order to be able to execute it in reasonable time. Please edit the script `exp_true_positives.py` accordingly (there are comments in the script to explain where) to reproduce Table 1.

The adastop library must be installed with `pip install .` from the root of the repository.