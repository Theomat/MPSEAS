from pseas.new_instance_selection.new_instance_selection import NewInstanceSelection
from pseas.model import Model

from typing import List

import numpy as np

from scipy.stats import wilcoxon

import warnings

class Oracle(NewInstanceSelection):

    def select(self, challenger_configuration: int, incumbent_configuration: int, perf_matrix: np.ndarray, perf_mask: np.ndarray, model: Model, predicted_perf_matrix: np.ndarray,  instance_features: np.ndarray) -> int:
        x1: List[float] = perf_matrix[perf_mask[:, incumbent_configuration], incumbent_configuration]
        x2: List[float] = perf_matrix[perf_mask[:, incumbent_configuration], challenger_configuration]

        selectables_instances = [i for i in range(perf_matrix.shape[0]) if not np.any(perf_mask[i, :])]
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)

            _, p_stop = wilcoxon(x1, x2, alternative="two-sided")
            best_choice = selectables_instances[0]
            best_confidence =  1 - p_stop


            for i in selectables_instances:
                new_x1 = x1 + [perf_matrix[i, challenger_configuration]]
                new_x2 = x2 + [perf_matrix[i, incumbent_configuration]]
                _, p_stop = wilcoxon(new_x1, new_x2, alternative="two-sided")
                if 1 - p_stop > best_confidence:
                    best_choice = i
                    best_confidence = 1 - p_stop
        return best_choice 



    def name(self) -> str:
        return "oracle"
