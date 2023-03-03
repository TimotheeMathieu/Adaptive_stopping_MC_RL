from ast import literal_eval
import copy
import functools
import glob
import json
import os
import pickle
import re
import subprocess

from joblib import Parallel, delayed
import numpy as np

from adastop.compare_agents import MultipleAgentsComparator

CONFIG_DIRNAME = "configs/"
COMP_CONFIG_FILENAME = "config.json"
COMP_FILENAME = "comparator.pkl"
RESULTS_FILENAME = "results.npy"


def get_agent_configs(dirpath):
    agent_regex = os.path.join(dirpath, "*.json")
    configs = []
    for agent_config_path in glob.glob(agent_regex):
        with open(agent_config_path, "r") as f:
            configs.append(json.load(f))
        assert 'train_cmd' in configs[-1], f"'train_cmd' must be specified in {agent_config_path}!"
        assert 'eval' in configs[-1], f"'eval' must be specified in {agent_config_path}!"
    return configs


def save(dirpath, comparator, config):
    # Make sure that the directory exists
    os.makedirs(dirpath, exist_ok=True)

    # Save the config
    with open(os.path.join(dirpath, COMP_CONFIG_FILENAME), "w") as f:
        json.dump(config, f)

    # Save the comparator
    with open(os.path.join(dirpath, COMP_FILENAME), "wb") as f:
        pickle.dump(comparator, f)


def load(dirpath, config):
    # Load the config
    with open(os.path.join(dirpath, COMP_CONFIG_FILENAME), "r") as f:
        config = json.load(f)

    # Load the comparator
    with open(os.path.join(dirpath, COMP_FILENAME), "rb") as f:
        comparator = pickle.load(f)

    return comparator, config


def init(dirpath, config):
    # Make sure that the directory exists
    assert os.path.isdir(dirpath), "'dir' must be a directory!"
    assert os.path.exists(dirpath), "'dir' must exist!"

    comparator = MultipleAgentsComparator(**config)
    save(dirpath, comparator, config) # save comparator and config
    return comparator, config


def run_batch(seeds, config, agent_idx, mode='train'):
    n_jobs = config.get('n_jobs', 1)

    assert mode in ['train', 'eval']

    # add args to command
    cmd = [config[f'{mode}_cmd']]
    args = config.get(f'{mode}_args', {})
    for k, v in args.items():
        cmd.extend([k, str(v)])

    # make the full command for each seed(with optional pre command)
    template = 'bash -c "{}"'
    seed_arg = config.get('seeding_option', '-s')
    pre_cmd = config.get('pre_cmd', None)

    cmds = []
    for seed in seeds:
        seed_cmd = copy.copy(cmd) + [seed_arg, str(seed)]
        if pre_cmd:
            seed_cmd = [pre_cmd + ';'] + seed_cmd
        cmds.append(template.format(" ".join(seed_cmd)))

    # execute all calls
    rets = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(subprocess.run)(c, shell=True, capture_output=True)
        for c in cmds)
    return agent_idx, rets


def parse_stdout(outputs, regex='AdaStop Evaluation[:]?[\s]*([-+]?[0-9]+\.?[0-9]*)'):
    return np.array(
        [float(re.search(regex, o.stdout.decode('utf-8')).group(1)) for o in outputs], dtype=np.float32)


def parse_files(dirpath, seeds, file_regex='adastop_{seed}.*'):
    results = []
    for seed in seeds:
        path_regex = os.path.join(dirpath, file_regex.replace('{seed}', str(seed)))
        matches = glob.glob(path_regex)
        assert len(matches) == 1, f"Found {len(matches)} matches for {path_regex}!"
        with open(matches[0], 'r') as f:
            results.append(float(f.read()))
    return np.array(results, dtype=np.float32)


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
        help='Significance level.')
    parser.add_argument('--beta', '-b', type=float, default=0.0,
        help='Power spent on early acceptance.')
    parser.add_argument('--comparisons', '-c', type=str, nargs="+", default=None,
        help='Comparisons to perform.')
    parser.add_argument('--seed', '-s', type=int, default=None,
        help='Random seed.')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    config = {
        "n": args.nb_fits,
        "K": args.K,
        "B": args.B,
        "alpha": args.alpha,
        "beta": args.beta,
        "comparisons": [literal_eval(t) for t in args.comparisons] if args.comparisons else None,
        "seed": args.seed,
    }
    dirpath = args.dir
    gen = np.random.default_rng(args.seed)

    # Make sure that the directory exists
    assert os.path.isdir(dirpath), "'dir' must be a directory!"
    assert os.path.exists(dirpath), "'dir' must exist!"

    # Load agent configs
    agent_configs = get_agent_configs(os.path.join(dirpath, CONFIG_DIRNAME))
    nb_agents = len(agent_configs)
    assert nb_agents > 0, "No agent config found in '{}'!".format(os.path.join(dirpath, CONFIG_DIRNAME))

    # Reset comparison if needed
    if args.reset:
        print('Resetting comparison.')
        if os.path.exists(os.path.join(dirpath, COMP_CONFIG_FILENAME)):
            os.remove(os.path.join(dirpath, COMP_CONFIG_FILENAME))
        if os.path.exists(os.path.join(dirpath, COMP_FILENAME)):
            os.remove(os.path.join(dirpath, COMP_FILENAME))
        if os.path.exists(os.path.join(dirpath, RESULTS_FILENAME)):
            os.remove(os.path.join(dirpath, RESULTS_FILENAME))

    # Initialize the comparator (or load it if it already exists)
    if not (os.path.exists(os.path.join(dirpath, COMP_CONFIG_FILENAME)) and 
            os.path.exists(os.path.join(dirpath, COMP_FILENAME))):
        print('Starting new comparison.')
        comparator, config = init(dirpath, config)
    else:
        print('Loading existing comparison.')
        comparator, config = load(dirpath, config)

    # Run comparison
    seeds = gen.integers(0, 2**32, size=(args.nb_fits * args.K, nb_agents))
    Z_path = os.path.join(dirpath, RESULTS_FILENAME)
    if os.path.exists(Z_path):
        Z = np.load(Z_path, allow_pickle=True).tolist()
    else:
        Z = [np.array([]) for _ in agent_configs]

    for k in range(args.K):
        print(f'Batch {k}:')
        # Run training
        # If the agent is still in one of the comparison considered, then run another batch of fits
        print('\ttraining agents...', end=' ')
        if k == 0:
            agents_to_train = list(range(nb_agents))
        else:
            agents_to_train = [i for i in range(nb_agents) if i in comparator.current_comparisons.ravel()]

        batch_args = []
        for i in agents_to_train:
            train_seeds = list(seeds[k * args.nb_fits: (k+1) * args.nb_fits, i].ravel())
            batch_args.append((train_seeds, agent_configs[i], i))
        
        train_output = Parallel(n_jobs=nb_agents, prefer="threads")(delayed(run_batch)(*args) for args in batch_args)
        train_output = {i: o for i, o in train_output}
        _validate_outputs(train_output.values())

        print('done.')

        # Run evaluations (if necessary)
        agents_to_eval = [i for i in agents_to_train if agent_configs[i].get('eval_cmd', None) is not None]
        if len(agents_to_eval) > 0:
            print('\tevaluating agents...', end=' ')

            batch_args = []
            for i in agents_to_eval:
                eval_seeds = list(seeds[k * args.nb_fits: (k+1) * args.nb_fits, i].ravel())
                batch_args.append((eval_seeds, agent_configs[i], i))

            eval_output = Parallel(n_jobs=nb_agents, prefer="threads")(delayed(run_batch)(*args, mode='eval') for args in batch_args)
            eval_output = {i: o for i, o in eval_output}
            _validate_outputs(eval_output.values())

            print('done.')
        
        # Collect evaluations
        print('\tcollecting results...', end=' ')
        for i in agents_to_train:
            if agent_configs[i]['eval'] == 'train_stdout':
                Z[i] = np.hstack([Z[i], parse_stdout(train_output[i])])
            elif agent_configs[i]['eval'] == 'eval_stdout':
                Z[i] = np.hstack([Z[i], parse_stdout(eval_output[i])])
            elif agent_configs[i].get('eval_dir', None) is not None:
                eval_dir = agent_configs[i]['eval_dir']
                eval_file_regex = agent_configs[i].get('eval_file_regex', 'adastop_{seed}.*')
                eval_seeds = list(seeds[k * args.nb_fits: (k+1) * args.nb_fits, i].ravel())
                Z[i] = np.hstack([Z[i], parse_files(eval_dir, eval_seeds, file_regex=eval_file_regex)])
            else:
                raise ValueError("Unknown evaluation method for agent {}!".format(i))
        print('done.')

        # Save batch results
        np.save(Z_path, Z)

        # Make early stopping if necessary
        decisions, T = comparator.partial_compare(Z, args.verbose)
        save(dirpath, comparator, config)

        counts = {}
        for d in comparator.decisions:
            counts[d] = counts.get(d, 0) + 1
        print('\tdecision summary: {} accept, {} reject, {} continue'.format(
            counts.get('accept', 0), counts.get('reject', 0), counts.get('continue', 0)))
        if np.all([d in ["accept", "reject"] for d in decisions]):
            break

    # Print final results
    print("Final results:")
    for i, decision in enumerate(comparator.decisions):
        print("Comparison {} : {}".format(comparator.comparisons[i], decision))
