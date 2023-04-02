from ast import literal_eval
import copy
import glob
import json
import logging
import os
import pickle
import re
import subprocess
import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from adastop.compare_agents import MultipleAgentsComparator


AGENT_DIRNAME = "agents/"
CMP_CONFIG_FILE = "config.json"
CMP_FILE = ".adastop_comparator.pkl"
LOGFILE = "adastop_log.txt"

logger = logging.getLogger(__name__)


def save(dirpath, comparator, config):
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, CMP_CONFIG_FILE), "w") as f:
        json.dump(config, f)
    with open(os.path.join(dirpath, CMP_FILE), "wb") as f:
        pickle.dump(comparator, f)


def load(dirpath):
    with open(os.path.join(dirpath, CMP_CONFIG_FILE), "r") as f:
        config = json.load(f)
    with open(os.path.join(dirpath, CMP_FILE), "rb") as f:
        comparator = pickle.load(f)
    return comparator, config


def init(dirpath, config):
    assert os.path.isdir(dirpath), "'dir' must be a directory!"
    assert os.path.exists(dirpath), "'dir' must exist!"

    comparator = MultipleAgentsComparator(**config)
    save(dirpath, comparator, config) # save comparator and config
    return comparator

    
def setup_logger(dirpath):
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(dirpath, LOGFILE))
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter('%(asctime)s | runner | %(levelname)s | %(message)s'))
    logger.addHandler(fh)


def get_agents(dirpath):
    agent_dir = os.path.join(dirpath, AGENT_DIRNAME)
    agent_regex = os.path.join(agent_dir, "*.sh")
    agent_scripts = glob.glob(agent_regex)
    return {os.path.basename(a).replace(".sh", ""): a for a in agent_scripts}


def save_evals(dirpath, k, evals_k):
    name = os.path.basename(os.path.normpath(dirpath))
    path = os.path.join(dirpath, f"{name}{k+1}.csv")
    df_k = {a: pd.Series(e) for a, e in evals_k.items()}
    df_k = pd.DataFrame(df_k)
    df_k.to_csv(path)


def run_batch(agent, seeds, script_path):
    cmds = [
        f'bash -c "{copy.copy(script_path)} {seed}"'
        for seed in seeds
    ]

    start = time.time()
    outputs = Parallel(prefer="processes")(
        delayed(subprocess.run)(c, shell=True, capture_output=True)
        for c in cmds)
    duration = time.time() - start

    logger.info(f"[{agent}] Ran batch of {len(seeds)} seeds in {duration:.5f}s.")
    return agent, outputs


def parse_stdout(outputs, regex='AdaStop Evaluation[:]?[\s]*([-+]?[0-9]+\.?[0-9]*)'):
    return np.array(
        [float(re.search(regex, o.stdout.decode('utf-8')).group(1)) for o in outputs], dtype=np.float32)


def _validate_outputs(outputs):
    for output_batch in outputs:
        for o in output_batch:
            if o.returncode != 0:
                raise RuntimeError(o.stderr.decode('utf-8'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str,
        help='Directory where the comparison will be saved.')
    parser.add_argument('--reset', '-r', action='store_true',
        help='Reset the comparison.')
    parser.add_argument('--nb-fits', '-n', type=int, default=5,
        help='Number of fits before each early stopping decision.')
    parser.add_argument('--K', '-K', type=int, default=6,
        help='Number of checks.')
    parser.add_argument('--B', '-B', type=int, default=10000,
        help='Number of random permutations to approximate the permutation distribution.')
    parser.add_argument('--alpha', '-a', type=float, default=0.05,
        help='Type I error.')
    parser.add_argument('--beta', '-b', type=float, default=0.0,
        help='Early accept parameter.')
    parser.add_argument('--comparisons', '-c', type=str, nargs="+", default=None,
        help='Comparisons to perform.')
    parser.add_argument('--seed', '-s', type=int, default=None,
        help='Random seed.')
    args = parser.parse_args()

    # Make sure that the directory exists
    dirpath = args.dir
    assert os.path.isdir(dirpath), "'dir' must be a directory!"
    assert os.path.exists(dirpath), "'dir' must exist!"

    cmp_path = os.path.join(dirpath, CMP_FILE)
    cmp_config_path = os.path.join(dirpath, CMP_CONFIG_FILE)

    # Reset comparison if needed
    if args.reset:
        print('Resetting comparison.')
        filenames = [CMP_CONFIG_FILE, CMP_FILE, LOGFILE]
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.exists(path):
                os.remove(path)

    # Setup logger
    setup_logger(dirpath)
    rng = np.random.default_rng(args.seed)

    # Load agent configs
    agents = get_agents(dirpath)

    agent_names, nb_agents = list(agents.keys()), len(agents)
    assert nb_agents > 0, "No agent config found in '{}'!".format(os.path.join(dirpath, AGENT_DIRNAME))

    print(f"Found {nb_agents} agents in '{dirpath}':")
    for i, name in enumerate(agent_names):
        print(f"  Agent {i}: {name}")
    print()

    # Get comparator and evaluation data
    if not (os.path.exists(cmp_config_path) and os.path.exists(cmp_path)):
        print('Starting new comparison.')

        config = {
            "n": args.nb_fits,
            "K": args.K,
            "B": args.B,
            "alpha": args.alpha,
            "beta": args.beta,
            "comparisons": [literal_eval(t) for t in args.comparisons] if args.comparisons else None,
            "seed": args.seed,
        }
        comparator = init(dirpath, config)

        data = {a: np.array([], dtype=np.float32) for a in agent_names}
        comparator.agent_names = agent_names
    else:
        print('Loading existing comparison.')
        comparator, config = load(dirpath)

        data = {a: comparator.eval_values[a] for a in agent_names}
        if max([len(v) for v in data.values()]) > comparator.K * args.n:
            raise ValueError('Error: you tried to use more groups than what was initially declared, this is not allowed by the theory.')
        assert "continue" in list(comparator.decisions.values()), "Test finished at last iteration."

    # Run comparison
    seeds = rng.integers(0, 2**32, size=(args.nb_fits * args.K, nb_agents))

    start_k = max([len(v) for v in data.values()]) // args.nb_fits
    for k in range(start_k, config['K']):
        print(f'Step {k}:')

        # Run training
        print('  training agents...', end=' ')

        if k == 0:
            agents_to_train = list(range(nb_agents))
        else:
            agents_to_train = [i for i in range(nb_agents) if i in comparator.current_comparisons.ravel()]

        batch_args = []
        for i in agents_to_train:
            agent_name = comparator.agent_names[i]
            batch_seeds = list(seeds[k * args.nb_fits: (k+1) * args.nb_fits, i].ravel())
            batch_args.append(
                (agent_name, batch_seeds, agents[agent_name]))
        
        batch_output = Parallel(n_jobs=nb_agents, prefer="threads")(delayed(run_batch)(*args) for args in batch_args)
        batch_output = {n: o for n, o in batch_output}
        _validate_outputs(batch_output.values())

        print('done.')
        
        # Collect evaluations
        print('  collecting results...', end=' ')
        evals_k = dict()
        for i in agents_to_train:
            a = comparator.agent_names[i]
            evals_k[a] = parse_stdout(batch_output[a])
            data[a] = np.hstack([data[a], evals_k[a]])
        print('done.')

        # Save batch results
        save_evals(dirpath, k, evals_k)

        # Run AdaStop comparison
        comparator.partial_compare(data, False)
        save(dirpath, comparator, config)

        # Print and log decision summary
        print('  current evaluations:')
        for name in agent_names:
            mean, std = np.mean(data[name]), np.std(data[name])
            print(f"    {name}: mean = {mean:.3f} +/- {std:.3f}")
            logger.info(f"[batch={k}] evaluation {name} = {mean:.6f} +/- {std:.6f}")
        
        print('  current decisions:')
        counts = {}
        for pair, decision in comparator.decisions.items():
            if decision in 'continue':
                counts['continue'] = counts.get('continue', 0) + 1
            else:
                counts['reject'] = counts.get('reject', 0) + 1

            i, j = tuple(map(int, pair[1:-1].split(' ')))
            name_i = comparator.agent_names[i]
            name_j = comparator.agent_names[j]

            print(f"    decision <{name_i}, {name_j}>: {decision}")
            logger.info(f"[batch={k}] comparison <{name_i}, {name_j}> = {decision}")

        print(f"  decision summary: {counts.get('reject', 0)} reject, {counts.get('continue', 0)} continue")

        if not "continue" in list(comparator.decisions.values()):
            break
