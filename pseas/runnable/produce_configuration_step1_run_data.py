"""
Run this script with -h for the help.

It produces for each method for a given dataset all the data needed to compare the methods on the specified dataset.
The strategies being compared are defined after line 88.
"""
from concurrent.futures import wait, ALL_COMPLETED
from concurrent.futures.process import ProcessPoolExecutor
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from pseas.test_env import ResetChoice, TestEnv
from pseas.strategy import Strategy
from pseas.runnable.strategy_comparator_helper import __evaluate__
from pseas.standard_strategy import StandardStrategy
from pseas.discrimination.subset_baseline import SubsetBaseline
from pseas.discrimination.wilcoxon import Wilcoxon
from pseas.instance_selection.random_baseline import RandomBaseline
from pseas.instance_selection.discrimination_based import DiscriminationBased
from pseas.instance_selection.udd import UDD
from pseas.instance_selection.variance_based import VarianceBased

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
    "ratio_instances": 1,
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
                             help=" (default: 5)"
                             )
argument_parser.add_argument('--ratio-instances',
                             type=int,
                             action='store',
                             default=argument_default_values['ratio_instances'],
                             help=" (default: 1)"
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

# =============================================================================
# Start Strategy Definition
# =============================================================================


discriminators = [
    # lambda: DistributionBased("cauchy", confidence=101),
    # add method using the model
    lambda: Wilcoxon(confidence=101),
    lambda: SubsetBaseline(.2)
]
selectors = [
    lambda: RandomBaseline(0),
    lambda: DiscriminationBased(1.2),
    lambda: UDD(alpha=0, beta=0), # Uncertainty sampling
    lambda: UDD(alpha=1, beta=1), # Find optimal alpha, beta?
    lambda: VarianceBased(),
    # lambda: InformationBased(), #TODO: try adaptation
]

strategy_makers = [
    lambda i, d: StandardStrategy(i, d),
]
# =============================================================================
# End Strategy Definition
# =============================================================================

# Check if file already exists
original_df: Optional[pd.DataFrame] = None
if os.path.exists(f"./runs{output_suffix}.csv"):
    original_df = pd.read_csv(f"./runs_{output_suffix}.csv")
    original_df = original_df.drop("Unnamed: 0", axis=1)
    print("Found existing data, continuing acquisition from save.")


def evaluate(scenario_path: str, strategy: Strategy, known_configs: np.ndarray, incumbent: int, instances: np.ndarray, seed: int = 0, target_confidence = .95) -> Tuple[Dict, Dict]:
    env = TestEnv(scenario_path, False, seed)

    challenger_list: List[int] = []
    for config in range(env.nconfigurations):
        if config not in known_configs:
            challenger_list.append(config)
            env.set_enabled(config, -1, False)
    
    for instance in range(env.ninstances):
        if instance not in instances:
            env.set_enabled(-1, instance, False)
            env.set_instance_count_for_eval(instance, False)

    env.rng.shuffle(challenger_list)
    env.fit_model()

    data: Dict[str, List] = {
        "time_used": [],
        "total_time": [],
        "current_challenger": [],
        "errors": [],
        "discarded": [],
        "total_challengers": []
    }

    metadata = {
        "incumbent": incumbent,
        "seed": seed,
        "strategy": strategy.name()
    }

    time_used: float = 0
    max_time_used: float = 0
    errors: int = 0
    discarded: int = 0
    challengers_total = 0

    while challenger_list:
        challengers_total += 1
        challenger = challenger_list.pop()
        state, information, _ = env.reset((incumbent, challenger))
        #print(type(strategy._instance_selection), flush=True)
        strategy.ready(**information)
        strategy.reset()
        strategy.feed(state)
        instances_done = 0

        max_time_used += env.current_challenger_max_total_time

        while instances_done < len(instances):
            state = env.step(strategy.choose_instance())
            strategy.feed(state)
            instances_done += 1
            # Add data            
            data["current_challenger"].append(challenger)
            data["time_used"].append(time_used + env.current_time)
            data["total_time"].append(max_time_used)
            data["errors"].append(errors)
            data["discarded"].append(discarded)
            data["total_challengers"].append(challengers_total)

            # If we predict our new configuration is not better with enough confidence we discard it
            if not strategy.is_better() and strategy.get_current_choice_confidence() >= target_confidence:
                discarded += 1
                break 

        env.choose(strategy.is_better())
        if env.is_challenger_better != strategy.is_better():
            errors += 1
        if strategy.is_better():
            break
        env.enable_from_last_run()
        time_used += env.current_time
    return metadata, data 






df: Dict = {
    "time_used": [],
    "total_time": [],
    "current_challenger": [],
    "errors": [],
    "discarded": [],
    "incumbent": [],
    "seed": [],
    "strategy": []
}

def callback(future):
    pbar.update(1)

    metadata, data = future.result()
    n = len(data["time_used"])
    for key, val in metadata.items():
        data[key] = [val] * n

    # Append to dataframe
    for key, val in data.items():
        df[key] += val

    # Save general dataframe
    if pbar.n % save_every == 0:
        df_tmp = pd.DataFrame(df)
        if original_df is not None:
            df_tmp = original_df.append(df_tmp)
        df_tmp.to_csv(f"./runs_{output_suffix}.csv")


def run(scenario_path, max_workers):
    print()
    env = TestEnv(scenario_path)

    # Select instances
    ninstances = round(ratio_instances * env.ninstances)
    selected_instances = env.rng.choice(list(range(env.ninstances)), size=ninstances)
    for instance in range(env.ninstances):
        if instance not in selected_instances:
            env.set_enabled(-1, instance, False)
            env.set_instance_count_for_eval(instance, False)

    # Subset of configurations
    known_configurations = env.rng.choice(list(range(env.nconfigurations)), size=nb_configurations)
    for config in range(env.nconfigurations):
        if config not in known_configurations:
            env.set_enabled(config, -1, False)

    # Get incumbent that is the fastest
    env.fit_model()
    incumbent = env.reset(ResetChoice.RESET_BEST)[1]["incumbent_configuration"]

    all_seeds = list(range(nb_seeds))

    # Generate strategies
    total: int = 0
    strategies: List[Tuple[Strategy, List[int]]] = []
    for discriminator in discriminators:
        for selection in selectors:
            for strategy_make in strategy_makers:
                strat = strategy_make(selection(), discriminator())
                if original_df is not None:
                    tmp = original_df[original_df["strategy"] == strat.name(
                    )]
                    seeds_done = list(tmp["seed"].unique())
                    if len(seeds_done) == 0:
                        continue           
                    total += nb_seeds - len(seeds_done)
                    strategies.append((strat, [x for x in all_seeds if x not in seeds_done]))
                else:
                    total += nb_seeds
                    strategies.append((strat, all_seeds))
    global pbar
    pbar = tqdm(total=total)
    futures = []
    executor = ProcessPoolExecutor(max_workers)
    for strategy, seeds in strategies:
        for seed in seeds:
            future = executor.submit(evaluate, scenario_path, strategy.clone(), known_configurations, incumbent, selected_instances, seed)
            future.add_done_callback(callback)
            futures.append(future)
    wait(futures, return_when=ALL_COMPLETED)


run(scenario_path, max_workers)
# Last save
df_tmp = pd.DataFrame(df)
if original_df is not None:
    df_tmp = original_df.append(df_tmp)
df_tmp.to_csv(f"./runs_{output_suffix}.csv")
