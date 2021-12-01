from pseas.new_instance_selection.new_instance_selection import NewInstanceSelection
from pseas.model import Model

import numpy as np
from typing import Callable, List
from scipy import optimize
def __compute_distance_matrix__(features: np.ndarray, distance: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """
    Computes the distance matrix between the instances.
    It assumes the distance function is symmetric that is d(x,y)=d(y,x) and it assumes d(x, x)=0.

    Parameters:
    -----------
    - features (np.ndarray) - the features of the instances
    - distance (Callable[[np.ndarray, np.ndarray], float]) - a function that given two features compute their distance

    Return:
    -----------
    The distance_matrix (np.ndarray) the distance matrix.
    """
    num_instances: int = features.shape[0]
    distance_matrix: np.ndarray = np.zeros(
        (num_instances, num_instances), dtype=np.float64)
    for instance1_index in range(num_instances):
        features1: np.ndarray = features[instance1_index]
        for instance2_index in range(instance1_index + 1, num_instances):
            d: float = distance(features1, features[instance2_index])
            distance_matrix[instance2_index, instance1_index] = d
            distance_matrix[instance1_index, instance2_index] = d
    return distance_matrix


def __find_weights__(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    instances: int = x.shape[0]
    features: int = x.shape[1]

    removed_instances = np.sum(mask <= 0)
    instances -= removed_instances

    qty: int = int(instances * (instances - 1) / 2)

    dx: np.ndarray = np.zeros((qty, features))
    dy: np.ndarray = np.zeros((qty,))

    # Compute dataset
    index: int = 0
    for i in range(instances):
        if mask[i] <= 0:
            continue
        for j in range(i + 1, instances):
            if mask[j] <= 0:
                continue
            dx[index] = x[i] - x[j]
            dy[index] = y[i] - y[j]
            index += 1
    np.square(dx, out=dx)
    np.abs(dy, out=dy)
    # np.square(dy, out=dy)

    # weights = argmin_w_i (norm [w_i (x_i -x'_i)]_i - |y - y'|)^2
    weights, residual = optimize.nnls(dx, dy)
    return np.sqrt(weights)

class UDD(NewInstanceSelection):

    def __init__(self, alpha: float = 1, beta: float = 1, k : int = 5) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.k : int = k

    def __uncertainty(self, perf_matrix: np.ndarray, selectables_instances, model: Model, challenger_configuration) -> List[int]:
        """
        Original: Difference between max vote and max second vote for classification
        
        Ours: variance of predictions among forest"""
        uncertainty: np.ndarray = np.zeros(perf_matrix.shape[0])
        for instance in selectables_instances:
            _, var = model.predict(challenger_configuration, instance)
            uncertainty[instance] = var
        return uncertainty

    def __k_nearest_neighbours(self, instance, selectables_instances, distances: np.ndarray):
        d = distances[instance, :]
        sorted = np.argsort(d)[::-1]
        k_best = []
        for i in sorted:
            if i in selectables_instances and i != instance:
                k_best.append(i)
                if len(k_best) == self.k:
                    break
        return k_best

    def __density(self, selectables_instances, distances: np.ndarray):
        densities = np.zeros(distances.shape[0], float)
        for instance in selectables_instances[:]:
            neighbours = self.__k_nearest_neighbours(instance, selectables_instances, distances)
            total: float = 0
            for neighbour in neighbours:
                dist: float = distances[instance, neighbour]
                total += dist*dist
            total /= max(1, len(neighbours))
            densities[instance] = total
        return densities


    def __diversity(self, selectables_instances, distances: np.ndarray):
        done_mask = np.array([i not in selectables_instances for i in range(distances.shape[0])])
        if np.any(done_mask):
            diversities = np.min(distances[:, done_mask], axis=1)
            diversities[done_mask] = 0
        else:
            diversities = np.zeros((len(selectables_instances)))
        return diversities


    def select(self, challenger_configuration: int, incumbent_configuration: int, perf_matrix: np.ndarray, perf_mask: np.ndarray, model: Model, predicted_perf_matrix: np.ndarray,  instance_features: np.ndarray) -> int:

        mask = np.sum(perf_mask, axis=1)
        # Find optimal distance function
        y = np.zeros((perf_matrix.shape[0]))
        for instance in range(y.shape[0]):
            if np.any(perf_mask[instance]):
                times = perf_matrix[instance, perf_mask[instance]]
                y[instance] = np.median(times)
        weights: np.ndarray = __find_weights__(instance_features, y, mask)
        distances = __compute_distance_matrix__(instance_features, lambda x1, x2: np.linalg.norm(weights * (x1 - x2)))

        selectables_instances = [i for i in range(perf_matrix.shape[0]) if not np.any(perf_mask[i, :])]
        current_configurations = np.any(perf_mask, axis=0)

        
        uncertainties = self.__uncertainty(perf_matrix, selectables_instances, model, challenger_configuration)
        # Normalize values in [0, 1]
        uncertainties -= np.min(uncertainties)
        uncertainties /= max(1e-3, np.max(uncertainties))
        if self.alpha == 0 and self.beta == 0:
            scores = uncertainties
        else:
            densities = self.__density(selectables_instances, distances)
            diversities = self.__diversity(selectables_instances, distances)
            # Normalize values in [0, 1]
            densities -= np.min(densities)
            diversities -= np.min(diversities)
            densities /= max(1e-3, np.max(densities))
            diversities /= max(1e-3, np.max(diversities))
            scores = uncertainties + self.alpha * densities - self.beta * diversities
        for i in range(perf_matrix.shape[0]):
            if i not in selectables_instances:
                scores[i] = 1e30
        return np.argmin(scores) 



    def name(self) -> str:
        return "uncertainty" if self.alpha == 0 and self.beta == 0 else "udd"
