# Auxiliary Scripts

## MuJoCo Comparisons

The script `plot_mujoco_comparisons.py` plots Adastop comparisons for the MuJoCo experiments.

This script assumes its given the path to a directory `parent_dir` containing the evaluations in `.npy` format, as follows:

```
parent_dir
- ant.npy
- halfcheetah.npy
- humanoid.npy
- humanoid.npy
- walker.npy
```

Additionally, there are more options to control the elements of the plot:

```
usage: plot_mujoco_comparisons.py [-h] --path PATH [--draw-table] [--draw-boxplot]

options:
  -h, --help      show this help message and exit
  --path PATH     Directory containing the evaluations.
  --draw-table    Whether to draw the `n_iter` table above decision table.
  --draw-boxplot  Whether to draw the boxplot below decision table.
```
