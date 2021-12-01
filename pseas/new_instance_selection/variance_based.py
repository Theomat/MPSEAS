from pseas.new_instance_selection.new_instance_selection import NewInstanceSelection
from pseas.model import Model

import numpy as np

class Variance(NewInstanceSelection):

    def select(self, challenger_configuration: int, incumbent_configuration: int, perf_matrix: np.ndarray, perf_mask: np.ndarray, model: Model, predicted_perf_matrix: np.ndarray,  instance_features: np.ndarray) -> int:

        selectables_instances = [i for i in range(perf_matrix.shape[0]) if not np.any(perf_mask[i, :])]
        current_configurations = np.any(perf_mask, axis=0)

        best_ratio = -1
        best_instance = 0
        for i in selectables_instances:
            std = np.std(predicted_perf_matrix[i, current_configurations])
            mean = np.median(predicted_perf_matrix[i, current_configurations])
            if std / mean > best_ratio:
                best_instance = i
                best_ratio = std / mean        
        
        return best_instance



    def name(self) -> str:
        return "variance"
