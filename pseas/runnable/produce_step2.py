"""
Run this script with -h for the help.

It produces for each method for a given dataset all the data needed to compare the methods on the specified dataset.
The strategies being compared are defined after line 88.
"""
from concurrent.futures import wait, ALL_COMPLETED
from concurrent.futures.process import ProcessPoolExecutor
import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from scipy.stats.morestats import wilcoxon
from tqdm import tqdm

from pseas.test_env import ResetChoice, TestEnv
from pseas.new_instance_selection.new_instance_selection import NewInstanceSelection
from pseas.new_instance_selection.random import Random
from pseas.new_instance_selection.oracle import Oracle
from pseas.new_instance_selection.variance_based import Variance
from pseas.new_instance_selection.discrimination_based import Discrimination
from pseas.new_instance_selection.udd import UDD

# =============================================================================
# Argument parsing.
# =============================================================================
import argparse
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Produce run data.")

argument_default_values: Dict = {
    "output_suffix": '',
    "save_every": 1,
    "max_workers": None,
    "scenario_path": './rundata/kissat_ibm',
    "nb_configurations": 50,
    "ratio_instances": .5,
    "nb_seeds": 5
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
                             help="Save data every X time. (default: 1)"
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
                             help=" (default: 50)"
                             )
argument_parser.add_argument('--nb-seeds',
                             type=int,
                             action='store',
                             default=argument_default_values['nb_seeds'],
                             help=" (default: 5)"
                             )
argument_parser.add_argument('--ratio-instances',
                             type=float,
                             action='store',
                             default=argument_default_values['ratio_instances'],
                             help=" (default: .5)"
                             )

parsed_parameters = argument_parser.parse_args()

nb_seeds: int = parsed_parameters.nb_seeds
save_every: int = parsed_parameters.save_every
max_workers: int = parsed_parameters.max_workers
scenario_path: str = parsed_parameters.scenario_path
nb_configurations: int = parsed_parameters.nb_configurations
ratio_instances: float = parsed_parameters.ratio_instances
output_suffix: str = scenario_path.strip('/').split('/')[-1]+'_'+str(nb_configurations)+'_'+str(ratio_instances)
# =============================================================================
# Finished parsing
# =============================================================================

TARGET_CONFIDENCE = .95
# =============================================================================
# Start Strategy Definition
# =============================================================================
selectors: List[Callable[[], NewInstanceSelection]] = [
    lambda: Random(0),
    # lambda: Oracle(),
    lambda: Variance(),
    # lambda: Discrimination(1.2),
    # lambda: UDD(0, 0),
    # lambda: UDD(1, 1)
]
# =============================================================================
# End Strategy Definition
# =============================================================================

# Check if file already exists
original_df_general: Optional[pd.DataFrame] = None
original_df_detailed: Optional[pd.DataFrame] = None
if os.path.exists(f"./selections_{output_suffix}.csv"):
    original_df_general = pd.read_csv(f"./selections_{output_suffix}.csv")
    original_df_general = original_df_general.drop("Unnamed: 0", axis=1)
    print("Found existing data, continuing acquisition from save.")


df_general = {
    "seed": [],
    "incumbent": [],
    "challenger": [],
    "selected_instances": [],
    "additional_time": [],
    "selection": []
}



def callback(future):
    pbar.update(1)

    stats = future.result()
    for k, v in stats.items():
        for el in v:
            df_general[k].append(el)

    # Save general dataframe
    if pbar.n % save_every == 0:
        df_tmp = pd.DataFrame(df_general)
        if original_df_general is not None:
            df_tmp = original_df_general.append(df_tmp)
        df_tmp.to_csv(f"./selections_{output_suffix}.csv")


def evaluate(scenario_path: str, selector: NewInstanceSelection, seed: int,
                 verbose: bool = False, **kwargs) -> Dict[str, Union[List[float], List[int], List[str]]]:
    env: TestEnv = TestEnv(scenario_path, verbose, seed=seed)


    # Select instances
    ninstances = round(ratio_instances * env.ninstances)
    selected_instances = env.rng.choice(list(range(env.ninstances)), size=ninstances)
    not_selected_instances: List[int] = []
    for instance in range(env.ninstances):
        if instance not in selected_instances:
            env.set_enabled(-1, instance, False)
            env.set_instance_count_for_eval(instance, False)
            not_selected_instances.append(instance)

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

    stats: Dict[str, Union[List[float], List[int], List[str]]] = {
        "seed": [],
        "incumbent": [],
        "challenger": [],
        "selected_instances": [],
        "additional_time": [],
        "selection": []
    }
    time_incumbents = env._results[selected_instances, incumbent].tolist()
    for challenger in challenger_list:
        # Get challenger times
        time_challenger = env._results[selected_instances, challenger].tolist()

        with warnings.catch_warnings():
            # warnings.simplefilter(action='ignore', category=UserWarning)
            _, p_stop = wilcoxon(time_incumbents, time_challenger, alternative="two-sided")
            confidence = 1 - p_stop
            added_instances: List[int] = []
            additional_time = 0
            incumbent_cpy = time_incumbents[:]
            info: Dict = {}
            while confidence < TARGET_CONFIDENCE and len(added_instances) < 10:
                if len(added_instances) == 0:
                    for inst in selected_instances:
                        env.set_enabled(challenger, inst, True)
                    _, info, _ = env.reset((incumbent, challenger))
                if added_instances:
                    env.set_enabled(incumbent, added_instances[-1], True)
                    env.set_enabled(challenger, added_instances[-1], True)
                    env.fit_model()
                    # Update perf
                    for inst in not_selected_instances:
                        for conf in known_configurations:
                            info["filled_perf"][inst, conf] = env._model.predict(conf, inst)[0]

                selected_instance = selector.select(challenger, incumbent, info["perf_matrix"], info["perf_mask"], info["model"], info["filled_perf"], info["features"])
                added_instances.append(selected_instance)
                assert selected_instance not in selected_instances
                additional_time += env._results[selected_instance, incumbent] + env._results[selected_instance, challenger]
                incumbent_cpy.append(env._results[selected_instance, incumbent])
                time_challenger.append(env._results[selected_instance, challenger])
                _, p_stop = wilcoxon(incumbent_cpy, time_challenger, alternative="two-sided")
                confidence = 1 - p_stop

            
            if len(added_instances) > 0:
                # re disable old instances
                for i in added_instances:
                    env.set_enabled(-1, i, False)
                env.set_enabled(-challenger, -1, False)
                stats["seed"].append(seed)
                stats["incumbent"].append(incumbent)
                stats["challenger"].append(challenger)
                stats["selected_instances"].append(len(added_instances))
                stats["additional_time"].append(additional_time)
                stats["selection"].append(selector.name())
            
    return stats

def run(scenario_path, max_workers):
    # Generate strategies
    total: int = 0
    strategies: List[Tuple[NewInstanceSelection, Dict]] = []
    for selection in selectors:
        selector = selection()
        seeds_done = []
        total += nb_seeds
        if original_df_general is not None:
            tmp = original_df_general[original_df_general["selection"] == selector.name(
            )]
            seeds_done = np.unique(
                tmp["seed"].values).tolist()
            total -= len(seeds_done)
        strategies.append([selection, seeds_done])

    global pbar 
    pbar = tqdm(total=total)
    futures = []
    executor = ProcessPoolExecutor(max_workers)
    for strategy, seeds_done in strategies:
        for seed in range(nb_seeds):
            if seed in seeds_done:
                continue
            future = executor.submit(evaluate, scenario_path, strategy(), seed)
            future.add_done_callback(callback)
            futures.append(future)
    wait(futures, return_when=ALL_COMPLETED)
    pbar.close()

TestEnv(scenario_path, True)
run(scenario_path, max_workers)
# Last save
df_tmp = pd.DataFrame(df_general)
if original_df_general is not None:
    df_tmp = original_df_general.append(df_tmp)
df_tmp.to_csv(f"./selections_{output_suffix}.csv")
