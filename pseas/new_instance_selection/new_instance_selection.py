from abc import ABC, abstractmethod

import numpy as np

from pseas.model import Model


class NewInstanceSelection(ABC):

    @abstractmethod
    def select(self, challenger_configuration: int, incumbent_configuration: int, perf_matrix: np.ndarray, perf_mask: np.ndarray, model: Model, predicted_perf_matrix: np.ndarray,  instance_features: np.ndarray) -> int:
        pass


    @abstractmethod
    def name(self) -> str:
        pass
