from enum import Enum
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np

import pseas.data.configuration_time_loader as configuration_extractor
from pseas.data.prior_information import fill_features, fit_rf_model

class ResetChoice(Enum):
    """Defines the type of reset for an environement."""

    RESET_BEST = 0
    """Compare against the best performing algorithm."""
    RESET_RANDOM = 1
    """Compare against a random algorithm."""

class TestEnv:
    """
    A test environment to measure the performance of a strategy.

    Parameters:
    -----------
    - seed (Optional[int]) - the seed to use. Default: None.
    """

    def __init__(
        self,
        scenario_path: str,
        verbose: bool = True,
        seed: Optional[int] = None
    ) -> None:
        self.rng: np.random.Generator = np.random.default_rng(seed)

        data, features, configurations = configuration_extractor.load_configuration_data(
            scenario_path
        )
        self.ninstances: int = data.shape[0]
        self.nconfigurations: int = data.shape[1]

        self._configurations: Dict[int, np.ndarray] = configurations
        self._features: np.ndarray = fill_features(features, self.ninstances)
        self._results: np.ndarray = data


        self._enabled: np.ndarray = np.ones_like(self._results, dtype=bool)
        self._evaluation_mask: np.ndarray = np.ones(self.ninstances, dtype=bool)

        if verbose:
            print(
                "Using",
                self.ninstances,
                "instances with",
                self.nconfigurations,
                "algorithms."
            )
        # stats
        self._correct: List[bool] = []
        self._time_ratio: List[float] = []
        self._choices: List[int] = []
        self._history: List[List] = []

    def __state__(self) -> Tuple[List[Optional[float]], List[float]]:
        times: List[Optional[float]] = [
            self._challenger_times[i] if self._done[i] else None
            for i in range(self.ninstances)
        ]
        times_cmp: List[float] = [
            self._incumbent_times[i] for i in range(self.ninstances)
        ]
        return times, times_cmp

    def reset(
        self, choice: Union[ResetChoice, Tuple[int, int]] = ResetChoice.RESET_RANDOM
    ) -> Tuple[Tuple[List[Optional[float]], List[float]], Dict, bool]:
        """
        Reset the current state of the environment.

        Parameters:
        -----------
        - choice (ResetChoice or (challenger_index, incumbent_index)) - the choice type of algorithm to be evaluating

        Return:
        -----------
        A tuple ((my_times, times_comparing), information, information_has_changed).
        my_times (List[Optional[float]]) is a list containing the times the algorithm took on the instances this algorithm was run on.
        If the algorithm wasn't run on a problem it is replaced by None.
        times_comparing (List[float]) is a list containing the times the algorithm we are comparing against took on the instances.
        information (Dict) is the data to pass to the ready function to the strategy
        """

        if isinstance(choice, ResetChoice):
            # Choose 2 algorithms
            if choice == ResetChoice.RESET_BEST:
                incumbent = np.argmin(np.sum(self._results[self._enabled], axis=0))
            evaluating = self.rng.choice([x for x in range(self.nconfigurations) if x != incumbent])
        else:
            incumbent, evaluating = choice

        filled_perf = np.copy(self._results)
        for i in range(self._results.shape[0]):
            for j in range(self._results.shape[1]):
                if not self._enabled[i, j]:
                    filled_perf[i, j] = self._model.predict(j, i)[0]

        information = {
            "perf_matrix": self._results,
            "perf_mask": self._enabled,
            "filled_perf": filled_perf,
            "features": self._features,
            "challenger_configuration": evaluating,
            "incumbent_configuration": incumbent,
            "model": self._model
        }
        self._challenger_times: np.ndarray = np.array([self._results[instance, evaluating] if self._evaluation_mask[instance] else 0
                                                       for instance in range(self.ninstances)])
        self._incumbent_times: np.ndarray = np.array([self._results[instance, incumbent] if self._evaluation_mask[instance] else 0
                                                      for instance in range(self.ninstances)])

        self._history.append([evaluating, incumbent, False])

        # Assign data
        self._done: np.ndarray = np.zeros(self.ninstances, dtype=np.bool_)
        return self.__state__(), information, True

    def fit_model(self):
        # Fit the model on the available data
        # All data is in self._results
        # And to check if instance, config is available check self._enabled[instance, config]
        masked_array: np.ndarray = np.copy(self._results)
        masked_array[np.logical_not(self._enabled)] = np.nan
        self._model = fit_rf_model(self._features, masked_array, self._configurations)
        #TODO: test

    def enable_from_last_run(self):
        last_challenger = self._history[-1][0]
        for instance, done in enumerate(self._done):
            if done:
                self.set_enabled(last_challenger, instance)
        #fit the RF completely again
        self.fit_model()
        #TODO: see how expensive this is, but partial updates are less accurate

    def set_enabled(self, configuration: int, instance: int, enabled: bool = True):
        """
        Add/Remove a pair (config, instance) of the dataset and act (even after reset) as if they were not in the dataset.
        Replace config or instance with -1 to act on all pairs for the configuration or the instance. (-1 acts like *)
        Must be done just before a reset to behave corretly.
        """
        if instance == -1:
            self._enabled[:, configuration] = enabled
        elif configuration == -1:
            self._enabled[instance, :] = enabled

    def set_instance_count_for_eval(self, instance: int, enabled: bool = True):
        """
        Add/Remove an instance of the dataset and act (even after reset) as if they were not in the dataset but only relative ot the comparison of two configurations (does not affect given data).
        Must be done just before a reset to behave corretly.
        """
        self._evaluation_mask[instance] = enabled

    def choose(self, better: bool):
        """
        Choose wether this algorithm is better or not than the one it's being compared to.
        Once you choose you should reset the environment.

        Parameters:
        -----------
        - better (bool) - indicates whether this algorithm is strictly better or not thant the one it's being compared to
        """
        self._correct.append(self.is_challenger_better == better)
        self._time_ratio.append(self.current_time / self.current_challenger_max_total_time)
        self._choices.append(np.sum(self._done))
        self._history[-1][-1] = better


    def step(self, instance: int) -> Tuple[List[Optional[float]], List[float]]:
        """
        Choose the next problem.

        Parameters:
        -----------
        - instance (int) - the instance on which to run the algorithm

        Return:
        ----------
        A tuple (my_times, times_comparing).
        my_times (List[float]) is a list containing the times the algorithm took on the instances this algorithm was run on.
        If the algorithm wasn't run on a problem it is replaced by None.
        times_comparing (List[float]) is a list containing the times the algorithm we are comparing against took on the instances.
        """
        assert not self._done[instance], f"Instance {instance} was already chosen!"
        self._done[instance] = True
        return self.__state__()

    @property
    def is_challenger_better(self) -> bool:
        """
        Return true iff the challenger is better than the incumbent.
        """
        return (
            np.sum(self._challenger_times)
            < np.sum(self._incumbent_times)
        )

    @property
    def current_time(self) -> float:
        """
        Total time used so far by the challenger.
        """
        return sum(
            [
                self._challenger_times[i]
                for i in range(self.ninstances)
                if self._done[i]
            ]
        )

    @property
    def current_instances(self) -> int:
        """
        Number of instances on which the challenger has been executed.
        """
        return np.sum(self._done)

    @property
    def current_incumbent_max_total_time(self) -> float:
        """
        Total time it would take to run the incumbent on all instances.
        """
        return np.sum(self._incumbent_times)

    @property
    def current_challenger_max_total_time(self) -> float:
        """
        Total time it would take to run the challenger on all instances.
        """
        return np.sum(self._challenger_times)

    def score(self, estimator: Callable[[Iterable], float]=np.median) -> float:
        correct = estimator(self._correct)
        time_ratio = estimator(self._time_ratio)
        return (correct - 0.5) * 2 * (1 - time_ratio)

    def stats(self, estimator=np.median) -> Tuple[float, float, float]:
        return (
            estimator(self._correct),
            estimator(self._time_ratio),
            estimator(self._choices),
        )

    def runs(
        self,
    ) -> Generator[Tuple[int, int, float, float, bool, bool, float, int], None, None]:
        for index, (challenger, incumbent, better) in enumerate(self._history):
            challenger_time: float = np.sum([self._results[instance, challenger] if self._evaluation_mask[instance] else 0 
                                              for instance in range(self.ninstances) ])
            incumbent_time: float = np.sum([self._results[instance, incumbent] if self._evaluation_mask[instance] else 0 
                                              for instance in range(self.ninstances) ])

            is_better: bool = challenger_time < incumbent_time

            yield challenger, incumbent, challenger_time, incumbent_time, is_better, better, self._time_ratio[
                index
            ], self._choices[index]

    def get_total_perf_matrix(self) -> np.ndarray:
        return self._results
