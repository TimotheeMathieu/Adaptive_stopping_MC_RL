import concurrent.futures
import functools
import multiprocessing
from typing import Optional
from rlberry.manager.evaluation import evaluate_agents, _get_last_xp
import pandas as pd
import rlberry
import numpy as np
import itertools
from scipy.stats import ttest_ind
from scipy import stats
import os

logger = rlberry.logger

def fit_stats(stats, save):
    stats.fit()
    if save:
        stats.save()
    return stats


class MultipleManagers:
    """
    Class to fit multiple AgentManager instances in parallel with multiple threads.

    Parameters
    ----------
    max_workers: int, default=None
        max number of workers (AgentManager instances) fitted at the same time.
    parallelization: {'thread', 'process'}, default: 'process'
        Whether to parallelize  agent training using threads or processes.
    mp_context: {'spawn', 'fork', 'forkserver'}, default: 'spawn'.
        Context for python multiprocessing module.
        Warning: If you're using JAX or PyTorch, it only works with 'spawn'.
                 If running code on a notebook or interpreter, use 'fork'.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        parallelization: str = "process",
        mp_context="spawn",
    ) -> None:
        super().__init__()
        self.instances = []
        self.max_workers = max_workers
        self.parallelization = parallelization
        self.mp_context = mp_context

    def append(self, agent_manager):
        """
        Append new AgentManager instance.

        Parameters
        ----------
        agent_manager : AgentManager
        """
        self.instances.append(agent_manager)

    def run(self, save=True):
        """
        Fit AgentManager instances in parallel.

        Parameters
        ----------
        save: bool, default: True
            If true, save AgentManager intances immediately after fitting.
            AgentManager.save() is called.
        """
        if self.parallelization == "thread":
            executor_class = concurrent.futures.ThreadPoolExecutor
        elif self.parallelization == "process":
            executor_class = functools.partial(
                concurrent.futures.ProcessPoolExecutor,
                mp_context=multiprocessing.get_context(self.mp_context),
            )
        else:
            raise ValueError(
                f"Invalid backend for parallelization: {self.parallelization}"
            )

        with executor_class(max_workers=self.max_workers) as executor:
            futures = []
            for inst in self.instances:
                futures.append(executor.submit(fit_stats, inst, save=save))

            fitted_instances = []
            for future in concurrent.futures.as_completed(futures):
                fitted_instances.append(future.result())

            self.instances = fitted_instances

    def save(self):
        """
        Pickle AgentManager instances and saves fit statistics in .csv files.
        The output folder is defined in each of the AgentManager instances.
        """
        for stats in self.instances:
            stats.save()

    def load(self, dirnames):
        for k, stats in enumerate(self.instances):
            agent_folder, dir_name = _get_last_xp(dirnames[k], stats.agent_name)
            last_xp_dirname = os.path.join(dir_name, agent_folder)
            self.instances[k] = stats.load(os.path.join( last_xp_dirname, 'manager_obj.pickle'))
            
    def stat_test(self, test_name="welch", alpha=0.05, n_eval=50):
        """
        Run statistical tests to assess the difference between the agents is sufficient to assert that they are different.
        The multimanager must have been run once with a sufficient amount of seed, the amount of seeds may be estimated using the
        :func:`pilot_study` method. Be careful that the same run cannot be use for both the pilot_study and stat_test or it would introduce bias.

        We use Holm's correction to account for multiple testing.

        Parameters
        ----------
        test_name: str, default="welch"
            Name of the test used.
        alpha: float, default=0.05
            Upper bound on Type I error of the test.
        n_eval: int, default=50
            Number of evaluations that we run to evaluate the effective mean reward of the trained agents.
        """

        eval_values = pd.DataFrame()
        managers = self.instances
        for manager in managers:
            logger.info("Evaluating Agent " + manager.agent_name)
            for idx in range(manager.n_fit):
                evaluation = np.mean(manager.eval_agents(n_eval, agent_id=idx))
                eval_values = pd.concat(
                    [
                        eval_values,
                        pd.DataFrame(
                            {
                                "agent_name": [manager.agent_name],
                                "n_simu": [idx],
                                "eval_value": [evaluation],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        agent_names = eval_values["agent_name"].unique()
        couples = itertools.combinations(agent_names, 2)

        results = pd.DataFrame()
        for a, b in couples:
            eval_a = eval_values.loc[eval_values["agent_name"] == a, "eval_value"]
            eval_b = eval_values.loc[eval_values["agent_name"] == b, "eval_value"]
            _, pval = ttest_ind(eval_a, eval_b, equal_var=False)
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            "agent1": [a],
                            "agent2": [b],
                            "p_value": [pval],
                            "eval_agent1": [np.mean(eval_a)],
                            "eval_agent2":[np.mean(eval_b)],
                        }
                    ),
                ],
                ignore_index=True,
            )
        ind_sort = np.argsort(results['p_value'].values)
        decisions = []
        results = results.iloc[ind_sort]
        for k,  p in enumerate(results['p_value'].values):
            if p < alpha/(len(results)-k):
                decisions.append('reject')
            else:
                decisions.append('accept')
        results['decisions']=decisions
        self.stat_test_results = results
        
    def pilot_study(self, alpha=0.05, beta=0.1, n_eval=50, maximum_n_seed=100):
        """
        Run a pilot study to estimate sample size needed.
        The multimanager must have been run once with a sufficient amount of seed to estimate the parameters of the problems (typically 5 seeds).
        
        We use Bonferroni correction to account for multiple testing. This is conservative.

        Parameters
        ----------
        alpha: float, default=0.05
            Upper bound on Type I error of the test.
        beta: float, default=0.1
            Upper bound on Type II error of the test.
        n_eval: int, default=50
            Number of evaluations that we run to evaluate the effective mean reward of the trained agents.
        maximum_n_seed: int, default=100
            Maximum number of seeds to consider.
        """
        eval_values = pd.DataFrame()
        managers = self.instances
        for manager in managers:
            logger.info("Evaluating Agent " + manager.agent_name)
            for idx in range(manager.n_fit):
                evaluation = np.mean(manager.eval_agents(n_eval, agent_id=idx))
                eval_values = pd.concat(
                    [
                        eval_values,
                        pd.DataFrame(
                            {
                                "agent_name": [manager.agent_name],
                                "n_simu": [idx],
                                "eval_value": [evaluation],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        agent_names = eval_values["agent_name"].unique()
        n_agents = len(agent_names)
        couples = itertools.combinations(agent_names, 2)

        candidate_n_seeds = np.arange(1, maximum_n_seed)
        needed_sample_size = 0
        for a, b in couples:
            eval_a = eval_values.loc[eval_values["agent_name"] == a, "eval_value"]
            eval_b = eval_values.loc[eval_values["agent_name"] == b, "eval_value"]            
            type_II = compute_beta(np.abs(np.mean(eval_a)-np.mean(eval_b)), candidate_n_seeds,
                                   alpha/(n_agents*(n_agents-1)/2),  s1=np.std(eval_a, ddof=1), s2 = np.std(eval_b, ddof=1))
            if np.any(type_II<beta):
                needed_sample_size = max(needed_sample_size, np.min(candidate_n_seeds[type_II < beta/(n_agents*(n_agents-1)/2)]))
            else:
                needed_sample_size = np.nan
        logger.info('The sample size needed is '+str(needed_sample_size))
        logger.warning("The same fits can't be used to do the test without incuring bias. please re-execute multimanager.run")
        return needed_sample_size
        
    @property
    def managers(self):
        return self.instances



def compute_beta(epsilon, sample_size, alpha=0.05, s1=None, s2=None):
    """
    Partially copied from the code of the article "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning" by
    Cedric Colas1, Olivier Sigaud and Pierre-Yves Oudeye (2018).
    
    Computes the probability of type-II error (or false positive rate) beta to detect and effect size epsilon
    when testing for a difference between performances of Algo1 versus Algo2, using a Welch's t-test
    with significance alpha and sample size N.
    Params
    ------
    - epsilon (int, float or list of int or float)
    The effect size one wants to be able to detect.
    - sample_size (int or list of int)
    The sample size (assumed equal for both algorithms).
    - alpha (float in ]0,1[)
    The significance level used by the Welch's t-test.
    - s1 (float)
    The standard deviation of Algo1, optional if data1 is provided.
    - s2 (float)
    The standard deviation of Algo2, optional if data2 is provided.
    """
    assert alpha < 1 and alpha > 0, "alpha must be in ]0,1["

    if type(sample_size) is int:
        sample_size = [sample_size]
        n_sample_size = 1
    else:
        n_sample_size = len(sample_size)

    s1 = s1
    s2 = s2

    results = np.zeros([n_sample_size])
    t_dist = stats.distributions.t

    for i_n, n in enumerate(sample_size):
        nu = (s1 ** 2 + s2 ** 2) ** 2 * (n - 1) / (s1 ** 4 + s2 ** 4)
        t_eps = epsilon / np.sqrt((s1 ** 2 + s2 ** 2) / n)
        t_crit = t_dist.ppf(1 - alpha, nu)
        results[i_n] = t_dist.cdf(t_crit - t_eps, nu)

    return results
