from pseas.new_instance_selection.new_instance_selection import NewInstanceSelection
from pseas.model import Model

from typing import Optional

import numpy as np


class Random(NewInstanceSelection):

    def __init__(self, seed: Optional[int] = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def select(self, challenger_configuration: int, incumbent_configuration: int, perf_matrix: np.ndarray, perf_mask: np.ndarray, model: Model, predicted_perf_matrix: np.ndarray,  instance_features: np.ndarray) -> int:
        selectables_instances = [i for i in range(perf_matrix.shape[0]) if not np.any(perf_mask[i, :])]
        return self._rng.choice(selectables_instances) 



    def name(self) -> str:
        return "random"
