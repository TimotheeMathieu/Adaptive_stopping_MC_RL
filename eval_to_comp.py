import json
import os
import pickle

import numpy as np

from adastop import MultipleAgentsComparator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(args.path), 'config.json')
    comparator_path = os.path.join(os.path.dirname(args.path), 'comparator.pkl')
    evals_path = os.path.join(os.path.dirname(args.path), 'results.npy')

    # Load configs from JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load evaluations from numpy file
    evals = np.load(evals_path, allow_pickle=True).item()

    # seed
    np.random.seed(config['seed'])

    # Instantiate comparator
    comparator = MultipleAgentsComparator(**config)

    # Create partial evaluations
    n, K = config['n'], config['K']
    partial_evals = []
    for k in range(K):
        aux = dict()
        for algo, v in evals.items():
            end = min((k + 1) * n, len(v))
            aux[algo] = v[:end]
        partial_evals.append(aux)

    # Compute partial comparisons
    for k in range(K):
        if max([len(v) for v in partial_evals[k].values()]) < (k + 1) * n:
            break
        if k > 1 and 'continue' not in comparator.decisions.values():
            break
        comparator.partial_compare(partial_evals[k])
        print('  current decisions:', comparator.decisions)

    # Plot
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(6,5))
    comparator.plot_results(axes=(ax1, ax2))
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(args.path), 'results.png'))

    # Save comparator
    with open(comparator_path, 'wb') as f:
        pickle.dump(comparator, f)
