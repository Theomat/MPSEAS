from pseas.instance_selection.instance_selection import InstanceSelection

from typing import Tuple, List, Optional

import numpy as np

class VarianceBased(InstanceSelection):
    """
    Variance based selection method.
    """

    def ready(self, filled_perf: np.ndarray, perf_mask: np.ndarray, **kwargs) -> None:

        self._scores = np.zeros((filled_perf.shape[0]))
        for instance in range(self._scores.shape[0]):
            if np.any(perf_mask[instance]):
                times = filled_perf[instance, perf_mask[instance]]
                self._scores[instance] = np.std(times) / np.median(times)
            else:
                self._scores[instance] = -1

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        not_run_mask: np.ndarray = np.array([time is None for time in state[0]])
        others = np.ones_like(self._scores) * -100
        others[not_run_mask] = self._scores[not_run_mask]
        self._next: int = np.argmax(others)

    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return "variance-based"

    def clone(self) -> 'VarianceBased':
        return VarianceBased()
