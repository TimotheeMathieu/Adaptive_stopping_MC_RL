# Auxiliary Scripts

## MuJoCo Comparisons

The script `plot_mujoco_comparisons.py` plots Adastop comparisons for the MuJoCo experiments.

This script assumes its given the path to a directory `parent_dir` containing a comparison directories for each environment:
```
parent_dir
- ant
  - ant1.csv
  - ...
- halfcheetah
  - halfcheetah1.csv
  - ...
...
```
The directory `experiments` in this repository contains all the information necessary to reproduce the AdaStop decision plots.

Additionally, there are more options to control the elements of the plot:

```
usage: plot_mujoco_comparisons.py [-h] --path PATH [--draw-table] [--draw-boxplot]

options:
  -h, --help      show this help message and exit
  --path PATH     Directory containing the evaluations.
  --draw-table    Whether to draw the `n_iter` table above decision table.
  --draw-boxplot  Whether to draw the boxplot below decision table.
```

## MuJoCo Sample Efficiency Plots

The script `plot_mujoco_sample_efficiency.py` plots sample efficiency plots for the MuJoCo experiments.

This script assumes its given the path to a directory `parent_dir` containing a comparison directories for each environment, as before.
Additionally, it requires that each environment directory contains a `aggregate_evals.csv` file, which contains the aggregated results for each algorithm
in different time steps.

The aggregate scores are computed from logs if they are not already present. For this, the script requires a `results` directory in each environment directory, containing the logs for each algorithm. For the main experiments, a version of the `experiments` directory containing all the logs in available in the supplementary material.
