"""
Run this script with -h for the help.

It produces for each method for a given dataset all the data needed to compare the methods on the specified dataset.
The strategies being compared are defined after line 88.
"""
from concurrent.futures import wait, ALL_COMPLETED
from concurrent.futures.process import ProcessPoolExecutor
import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from pseas.instance_selection.instance_selection import InstanceSelection
from tqdm import tqdm

from pseas.test_env import ResetChoice, TestEnv
from pseas.strategy import Strategy
from pseas.standard_strategy import StandardStrategy
from pseas.discrimination.subset_baseline import SubsetBaseline
from pseas.discrimination.wilcoxon import Wilcoxon
from pseas.instance_selection.random_baseline import RandomBaseline
from pseas.instance_selection.discrimination_based import DiscriminationBased
from pseas.instance_selection.variance_based import VarianceBased
from pseas.instance_selection.information_based import InformationBased
from pseas.instance_selection.udd import UDD

# =============================================================================
# Argument parsing.
# =============================================================================
import argparse
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Produce run data.")

argument_default_values: Dict = {
    "output_suffix": '',
    "save_every": 5,
    "max_workers": None,
    "scenario_path": './rundata/kissat_ibm',
    "nb_configurations": 10,
    "ratio_instances": .1,
    "nb_seeds": 10
}
argument_parser.add_argument('-o', '--output-suffix',
                             type=str,
                             action='store',
                             default=argument_default_values['output_suffix'],
                             help="CSV data filename suffix (default: '[scenario]_[nb configurations]_[ratio instance]')"
                             )
argument_parser.add_argument('--save-every',
                             type=int,
                             action='store',
                             default=argument_default_values['save_every'],
                             help="Save data every X time. (default: 5)"
                             )
argument_parser.add_argument('--max-workers',
                             type=int,
                             action='store',
                             default=argument_default_values['max_workers'],
                             help="Max number of processes. (default: None)"
                             )
argument_parser.add_argument('--scenario-path',
                             type=str,
                             action='store',
                             default=argument_default_values['scenario_path'],
                             help=" (default: './rundata/kissat_ibm')"
                             )
argument_parser.add_argument('--nb-configurations',
                             type=int,
                             action='store',
                             default=argument_default_values['nb_configurations'],
                             help=" (default: 10)"
                             )
argument_parser.add_argument('--nb-seeds',
                             type=int,
                             action='store',
                             default=argument_default_values['nb_seeds'],
                             help=" (default: 10)"
                             )
argument_parser.add_argument('--ratio-instances',
                             type=float,
                             action='store',
                             default=argument_default_values['ratio_instances'],
                             help=" (default: 1)"
                             )
argument_parser.add_argument('--disc',
                             action='store_true',
                             help=" (default: False) instaed of GridSearch for UDD do it for discrimination"
                             )
parsed_parameters = argument_parser.parse_args()

nb_seeds: int = parsed_parameters.nb_seeds
save_every: int = parsed_parameters.save_every
max_workers: int = parsed_parameters.max_workers
scenario_path: str = parsed_parameters.scenario_path
nb_configurations: int = parsed_parameters.nb_configurations
ratio_instances: float = parsed_parameters.ratio_instances
disc_instead_udd: bool = parsed_parameters.disc
name: str = "discrimination" if disc_instead_udd else "udd"
output_suffix: str = scenario_path.strip('/').split('/')[-1]+'_'+str(nb_configurations)+'_'+str(ratio_instances)+"_"+name
# =============================================================================
# Finished parsing
# =============================================================================


# =============================================================================
# Start Strategy Definition
# =============================================================================


discriminators = [
    lambda: Wilcoxon(confidence=101),
]
selectors: List[Callable[[], InstanceSelection]] = []
if not disc_instead_udd:
    parameters_1 = np.linspace(.2, 2, num=10).tolist()
    parameters_2 = np.linspace(.2, 2, num=10).tolist()
    selectors = [UDD(p1, p2) for p1 in parameters_1 for p2 in parameters_2]
else:
    parameters = np.linspace(1.01, 2, num=10).tolist()
    selectors = [DiscriminationBased(p) for p in parameters]
strategy_makers = [
    lambda i, d: StandardStrategy(i, d),
]
# =============================================================================
# End Strategy Definition
# =============================================================================

# Check if file already exists
original_df_general: Optional[pd.DataFrame] = None
original_df_detailed: Optional[pd.DataFrame] = None
if os.path.exists(f"./runs_{output_suffix}.csv"):
    original_df_general = pd.read_csv(f"./runs_{output_suffix}.csv")
    original_df_general = original_df_general.drop("Unnamed: 0", axis=1)

    original_df_detailed = pd.read_csv(
        f"./detailed_runs_{output_suffix}.csv")
    original_df_detailed = original_df_detailed.drop(
        "Unnamed: 0", axis=1)
    print("Found existing data, continuing acquisition from save.")


df_general = {
    "y_true": [],
    "y_pred": [],
    "time": [],
    "perf_eval": [],
    "perf_cmp": [],
    "instances": [],
    "strategy": [],
    "a_new": [],
    "a_old": [],
    "seed": []
}


df_detailed = {
    "strategy": [],
    "confidence": [],
    "time": [],
    "instances": [],
    "prediction": [],
    "a_new": [],
    "a_old": [],
    "seed": []
}


def callback(future):
    pbar.update(1)

    strat_name, runs, dico = future.result()
    # Fill detailed dataframe
    stats = dico["stats"]
    for k, v in stats.items():
        for el in v:
            df_detailed[k].append(el)

    # Save detailed dataframe
    if pbar.n % save_every == 0:
        df_tmp = pd.DataFrame(df_detailed)
        if original_df_detailed is not None:
            df_tmp = original_df_detailed.append(df_tmp)
        df_tmp.to_csv(f"./detailed_runs_{output_suffix}.csv")

    # real data
    real = dico["real"]
    challengers: List[int] = real["challenger"]

    seed = stats["seed"][-1]

    # Fill general dataframe
    for challenger, incumbent, perf_chall, perf_inc, y_true, _, _, _ in runs:
        df_general["y_true"].append(y_true)
        df_general["perf_eval"].append(perf_chall)
        df_general["perf_cmp"].append(perf_inc)
        df_general["strategy"].append(strat_name)
        df_general["a_new"].append(challenger)
        df_general["a_old"].append(incumbent)
        index = challengers.index(challenger)
        df_general["time"].append(real["time"][index])
        df_general["instances"].append(real["instances"][index])
        df_general["y_pred"].append(real["prediction"][index])
        df_general["seed"].append(seed)
    # Save general dataframe
    if pbar.n % save_every == 0:
        df_tmp = pd.DataFrame(df_general)
        if original_df_general is not None:
            df_tmp = original_df_general.append(df_tmp)
        df_tmp.to_csv(f"./runs_{output_suffix}.csv")


def evaluate(scenario_path: str, strategy: Strategy, seed: int,
                 verbose: bool = False, **kwargs) -> Tuple[str, List[Tuple[int, int, float, float, bool, bool, float, int]], Dict]:
    env: TestEnv = TestEnv(scenario_path, verbose, seed=seed)


    # Select instances
    ninstances = round(ratio_instances * env.ninstances)
    selected_instances = env.rng.choice(list(range(env.ninstances)), size=ninstances)
    for instance in range(env.ninstances):
        if instance not in selected_instances:
            env.set_enabled(-1, instance, False)
            env.set_instance_count_for_eval(instance, False)

    # Subset of configurations
    known_configurations = env.rng.choice(list(range(env.nconfigurations)), size=nb_configurations)
    challenger_list: List[int] = []
    for config in range(env.nconfigurations):
        if config not in known_configurations:
            env.set_enabled(config, -1, False)
            challenger_list.append(config)

    # Get incumbent that is the fastest
    env.fit_model()
    incumbent: int = env.reset(ResetChoice.RESET_BEST)[1]["incumbent_configuration"]
    env._history.clear()

    stats = {
            "time": [],
            "confidence": [],
            "prediction": [],
            "strategy": [],
            "a_new": [],
            "a_old": [],
            "instances": [],
            "seed": []
    }
    real = {
        "prediction": [],
        "time": [],
        "challenger": [],
        "instances": [],
    }
    to_ratio = lambda x: int(np.floor(x * 100))
    label: str = strategy.name()
    for challenger in challenger_list:
        state, information, information_has_changed = env.reset((incumbent, challenger))
        if information_has_changed:
            strategy.ready(**information)
        strategy.reset()
        strategy.feed(state)
        last_time_ratio: float = 0
        instances : int = 0
        finished: bool = False
        while instances < env.ninstances:
            state = env.step(strategy.choose_instance())
            strategy.feed(state)
            instances += 1
            #  Update if time changed enough
            time_ratio: float = env.current_time / env.current_challenger_max_total_time
            if to_ratio(last_time_ratio) < to_ratio(time_ratio):
                for i in range(to_ratio(last_time_ratio), to_ratio(time_ratio)):
                        # Update predictions
                        stats["time"].append(i)
                        stats["prediction"].append(
                            strategy.is_better() == env.is_challenger_better)
                        stats["strategy"].append(label)
                        stats["a_new"].append(challenger)
                        stats["a_old"].append(incumbent)
                        stats["instances"].append(instances)
                        stats["seed"].append(seed)

                        # Update confidence
                        try:
                            stats["confidence"].append(
                                strategy.get_current_choice_confidence() * 100)
                        except AttributeError:
                            stats["confidence"].append(100)
                last_time_ratio = time_ratio

            if not finished and strategy.get_current_choice_confidence() >= .95:
                if isinstance(strategy._discrimination, Wilcoxon) and env.current_instances < 5:
                    continue
                finished = True
                real["challenger"].append(challenger)
                real["prediction"].append(strategy.is_better())
                real["time"].append(env.current_time / env.current_challenger_max_total_time)
                real["instances"].append(env.current_instances)
        env.choose(strategy.is_better())
        # Fill in the rest
        for i in range(to_ratio(last_time_ratio), 101):
                # Update predictions
                stats["time"].append(i)
                stats["strategy"].append(label)
                stats["a_new"].append(challenger)
                stats["a_old"].append(incumbent)
                stats["instances"].append(instances)
                stats["prediction"].append(
                    strategy.is_better() == env.is_challenger_better)
                stats["seed"].append(seed)

                # Update confidence
                try:
                    stats["confidence"].append(
                        strategy.get_current_choice_confidence() * 100)
                except AttributeError:
                    stats["confidence"].append(100)
   
        if not finished:
            finished = True
            real["challenger"].append(challenger)
            real["prediction"].append(strategy.is_better())
            real["time"].append(env.current_time / env.current_challenger_max_total_time)
            real["instances"].append(env.current_instances)
    kwargs["stats"] = stats
    kwargs["real"] = real
    kwargs["a_old"] = incumbent
    return strategy.name(), list(env.runs()), kwargs

def run(scenario_path, max_workers):
    print()
    env = TestEnv(scenario_path)
    
    n_algos = env.nconfigurations
    # Generate strategies
    total: int = 0
    strategies: List[Tuple[Strategy, Dict]] = []
    for discriminator in discriminators:
        for selection in selectors:
            for strategy_make in strategy_makers:
                strat = strategy_make(selection, discriminator())
                seeds_done = []
                total += nb_seeds
                if original_df_general is not None:
                    tmp = original_df_general[original_df_general["strategy"] == strat.name(
                    )]
                    seeds_done = np.unique(
                        tmp["seed"].values).tolist()
                    total -= len(seeds_done)
                strategies.append([strat, seeds_done])

    global pbar 
    pbar = tqdm(total=total)
    futures = []
    executor = ProcessPoolExecutor(max_workers)
    for strategy, seeds_done in strategies:
        for seed in range(nb_seeds):
            if seed in seeds_done:
                continue
            future = executor.submit(evaluate, scenario_path, strategy.clone(), seed)
            future.add_done_callback(callback)
            futures.append(future)
    wait(futures, return_when=ALL_COMPLETED)
    pbar.close()


run(scenario_path, max_workers)
# Last save
df_tmp = pd.DataFrame(df_detailed)
if original_df_detailed is not None:
    df_tmp = original_df_detailed.append(df_tmp)
df_tmp.to_csv(f"./detailed_runs_{output_suffix}.csv")
df_tmp = pd.DataFrame(df_general)
if original_df_general is not None:
    df_tmp = original_df_general.append(df_tmp)
df_tmp.to_csv(f"./runs_{output_suffix}.csv")
