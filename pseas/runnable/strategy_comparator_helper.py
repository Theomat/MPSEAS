"""
Thsi is only a helper to generate data.
It enables to collect data while running a specific strategy.
"""

from numpy import floor
from pseas.test_env import TestEnv
from pseas.strategy import Strategy
from pseas.discrimination.wilcoxon import Wilcoxon

from typing import Callable, List, Dict, Optional, Tuple, Union

from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import wait, ALL_COMPLETED


def __evaluate__(env: TestEnv, strategy: Strategy, incumbent: int, challenger_list: List[int], **kwargs) -> Tuple[Strategy, TestEnv, Dict]:
    """
    For one strategy evaluate for all pairs (incumbent, challenger) where challenger is in challenger_list.
    """
    stats: Dict[str, List] = {
            "time": [],
            "confidence": [],
            "prediction": [],
            "strategy": [],
            "a_new": [],
            "a_old": [],
            "instances": []
    }
    real: Dict[str, List] = {
        "prediction": [],
        "time": [],
        "a_old": [],
        "instances": []
    }
    to_ratio = lambda x: int(floor(x * 100))
    label: str = strategy.name()
    for challenger in challenger_list:
        if challenger == incumbent:
            continue
        state, information, should_call_ready = env.reset((challenger, incumbent))
        if should_call_ready:
            strategy.ready(**information)
        strategy.reset()
        strategy.feed(state)
        last_time_ratio: float = 0
        instances : int = 0
        finished: bool = False
        while instances < env._n_instances:
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
                real["a_old"].append(incumbent)
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
                # Update confidence
                try:
                    stats["confidence"].append(
                        strategy.get_current_choice_confidence() * 100)
                except AttributeError:
                    stats["confidence"].append(100)
   
        if not finished:
            finished = True
            real["a_old"].append(incumbent)
            real["prediction"].append(strategy.is_better())
            real["time"].append(env.current_time / env.current_challenger_max_total_time)
            real["instances"].append(env.current_instances)
    kwargs["stats"] = stats
    kwargs["real"] = real
    return strategy, env, kwargs

