from pseas.instance_selection.instance_selection import InstanceSelection

from typing import Tuple, List, Optional

import numpy as np

class VarianceBased(InstanceSelection):
    """
    Variance based selection method.
    """

    def ready(self, filled_perf: np.ndarray, **kwargs) -> None:
        locs = np.median(filled_perf, axis=1)
        scales = np.std(filled_perf, axis=1)
        self._scores: np.ndarray = scales / np.maximum(locs, 1)

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
