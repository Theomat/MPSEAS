"""
Run this script with -h for the help.

It produces for each method for a given dataset all the data needed to compare the methods on the specified dataset.
The strategies being compared are defined after line 88.
"""
from concurrent.futures import wait, ALL_COMPLETED
from concurrent.futures.process import ProcessPoolExecutor
import os
from typing import Dict, List, Optional

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
    "nb_configurations":10,
    "ratio_instances":1,
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
argument_parser.add_argument('--ratio-instances',
                             type=int,
                             action='store',
                             default=argument_default_values['ratio_instances'],
                             help=" (default: 1)"
                             )

parsed_parameters = argument_parser.parse_args()


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
original_df_general: Optional[pd.DataFrame] = None
original_df_detailed: Optional[pd.DataFrame] = None
if os.path.exists(f"./runs{output_suffix}.csv"):
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
    "perf_challenger": [],
    "perf_incumbent": [],
    "instances": [],
    "strategy": [],
    "a_new": [],
    "a_old": [],
    "dataset": [],
}


df_detailed = {
    "strategy": [],
    "confidence": [],
    "time": [],
    "instances": [],
    "prediction": [],
    "a_new": [],
    "a_old": [],
    "dataset": [],
}

pbar = tqdm(total=0)


def callback(future):
    pbar.update(1)

    strat, env, dico = future.result()

    # Fill detailed dataframe
    stats = dico["stats"]
    for k, v in stats.items():
        if k == "strategy":
            for el in v:
                df_detailed["dataset"].append(dico["dataset"])
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

    # Fill general dataframe
    for challenger, incumbent, perf_challenger, perf_incumbent, y_true, _, _, _ in env.runs():
        df_general["y_true"].append(y_true)
        df_general["perf_challenger"].append(perf_challenger)
        df_general["perf_incumbent"].append(perf_incumbent)
        df_general["strategy"].append(strat.name())
        df_general["a_new"].append(challenger)
        df_general["a_old"].append(incumbent)
        df_general["dataset"].append(dico["dataset"])
        df_general["time"].append(real["time"][incumbent])
        df_general["instances"].append(real["instances"][incumbent])
        df_general["y_pred"].append(real["prediction"][incumbent])
    # Save general dataframe
    if pbar.n % save_every == 0:
        df_tmp = pd.DataFrame(df_general)
        if original_df_general is not None:
            df_tmp = original_df_general.append(df_tmp)
        df_tmp.to_csv(f"./runs_{output_suffix}.csv")


def run(scenario_path, max_workers):
    print()
    env = TestEnv(scenario_path)
    dataset_name: str = scenario_path[scenario_path.rfind("/")+1:]

    # Select instances
    ninstances = round(ratio_instances * env.ninstances)
    selected_instances = env._generator.choice(list(range(env.ninstances)), size=ninstances)
    for instance in range(env.ninstances):
        if instance not in selected_instances:
            env.set_enabled(-1, instance, False)
            env.set_instance_count_for_eval(instance, False)

    # Subset of configurations
    challenger_list = env._generator.choice(list(range(env.nconfigurations)), size=nb_configurations)
    for config in range(env.nconfigurations):
        if config not in challenger_list:
            env.set_enabled(config, -1, False)

    # Get incumbent that is the fastest
    incumbent = env.reset(ResetChoice.RESET_BEST)[1]["incumbent_configuration"]

    # Generate strategies
    strategies: List[Strategy] = []
    for discriminator in discriminators:
        for selection in selectors:
            for strategy_make in strategy_makers:
                strat = strategy_make(selection(), discriminator())
                if original_df_general is not None:
                    tmp = original_df_general[original_df_general["strategy"] == strat.name(
                    )]
                    tmp = tmp[tmp["dataset"] == dataset_name]
                    if tmp.shape[0] > 0:
                        continue            
                strategies.append(strat)
    pbar.total = len(strategies)
    executor = ProcessPoolExecutor(max_workers)
    futures = []
    for strategy in strategies:
        new_env = TestEnv(scenario_path, False, 0)
        new_env._enabled = env._enabled
        new_env._evaluation_mask = env._evaluation_mask
        future = executor.submit(__evaluate__, env, strategy, incumbent, challenger_list)
        future.add_done_callback(callback)
        futures.append(future) 
    wait(futures, return_when=ALL_COMPLETED)


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
