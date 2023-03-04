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

from adastop.compare_agents import MultipleAgentsComparator


CONFIG_DIRNAME = "configs/"
COMP_CONFIG_FILENAME = "config.json"
COMP_FILENAME = "comparator.pkl"
RESULTS_FILENAME = "results.npy"
LOGFILE = "log.txt"

logger = logging.getLogger(__name__)


def setup_logger(dirpath):
    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(dirpath, LOGFILE))
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s | runner | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    #logger.addHandler(ch)


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
    start = time.time()
    rets = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(subprocess.run)(c, shell=True, capture_output=True)
        for c in cmds)
    duration = time.time() - start

    logger.info(f"[{config['name']}] Ran {mode} batch of {len(seeds)} seeds in {duration:.5f}s.")
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

    # Reset comparison if needed
    if args.reset:
        print('Resetting comparison.')
        if os.path.exists(os.path.join(dirpath, COMP_CONFIG_FILENAME)):
            os.remove(os.path.join(dirpath, COMP_CONFIG_FILENAME))
        if os.path.exists(os.path.join(dirpath, COMP_FILENAME)):
            os.remove(os.path.join(dirpath, COMP_FILENAME))
        if os.path.exists(os.path.join(dirpath, RESULTS_FILENAME)):
            os.remove(os.path.join(dirpath, RESULTS_FILENAME))
        if os.path.exists(os.path.join(dirpath, LOGFILE)):
            os.remove(os.path.join(dirpath, LOGFILE))

    # Make sure that the directory exists
    assert os.path.isdir(dirpath), "'dir' must be a directory!"
    assert os.path.exists(dirpath), "'dir' must exist!"

    # Setup logger
    setup_logger(dirpath)
    gen = np.random.default_rng(args.seed)

    # Load agent configs
    agent_configs = get_agent_configs(os.path.join(dirpath, CONFIG_DIRNAME))
    agent_names = [ac['name'] for ac in agent_configs]
    nb_agents = len(agent_configs)
    assert nb_agents > 0, "No agent config found in '{}'!".format(os.path.join(dirpath, CONFIG_DIRNAME))

    print(f"Found {nb_agents} agents in '{dirpath}':")
    for i, ac in enumerate(agent_configs):
        print(f"  Agent {i}: {ac['name']}")
    print()

    

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
    eval_values_path = os.path.join(dirpath, RESULTS_FILENAME)
    if os.path.exists(eval_values_path):
        eval_values = np.load(eval_values_path, allow_pickle=True).tolist()
    else:
        eval_values = {name: np.array([]) for name in agent_names}

    for k in range(args.K):
        print(f'Step {k}:')

        # Run training
        # If the agent is still in one of the comparison considered, then run another batch of fits
        print('  training agents...', end=' ')

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
            print('  evaluating agents...', end=' ')

            batch_args = []
            for i in agents_to_eval:
                eval_seeds = list(seeds[k * args.nb_fits: (k+1) * args.nb_fits, i].ravel())
                batch_args.append((eval_seeds, agent_configs[i], i))

            eval_output = Parallel(n_jobs=nb_agents, prefer="threads")(delayed(run_batch)(*args, mode='eval') for args in batch_args)
            eval_output = {i: o for i, o in eval_output}
            _validate_outputs(eval_output.values())

            print('done.')
        
        # Collect evaluations
        print('  collecting results...', end=' ')
        for i in agents_to_train:
            name, eval_type = agent_configs[i]['name'], agent_configs[i]['eval']
            if eval_type == 'train_stdout':
                eval_values[name] = np.hstack([eval_values[name], parse_stdout(train_output[i])])
            elif eval_type == 'eval_stdout':
                eval_values[name] = np.hstack([eval_values[name], parse_stdout(eval_output[i])])
            elif agent_configs[i].get('eval_dir', None) is not None:
                eval_dir = agent_configs[i]['eval_dir']
                eval_file_regex = agent_configs[i].get('eval_file_regex', 'adastop_{seed}.*')
                eval_seeds = list(seeds[k * args.nb_fits: (k+1) * args.nb_fits, i].ravel())
                eval_values[name] = np.hstack([eval_values[name], parse_files(eval_dir, eval_seeds, file_regex=eval_file_regex)])
            else:
                raise ValueError(f"Unknown evaluation method for agent {i}!")
        print('done.')

        # Save batch results
        np.save(eval_values_path, eval_values)

        # Make early stopping if necessary
        comparator.partial_compare(eval_values, args.verbose)
        save(dirpath, comparator, config)

        # Print and log decision summary
        print('  current evaluations:')
        for name in agent_names:
            mean, std = np.mean(eval_values[name]), np.std(eval_values[name])
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
            name_i, name_j = agent_configs[i]['name'], agent_configs[j]['name']

            print(f"    decision <{name_i}, {name_j}>: {decision}")
            logger.info(f"[batch={k}] comparison <{name_i}, {name_j}> = {decision}")

        print(f"  decision summary: {counts.get('reject', 0)} reject, {counts.get('continue', 0)} continue")

        if np.all([d != 'continue' for d in comparator.decisions.values()]):
            break
