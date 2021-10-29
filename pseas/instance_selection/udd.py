from typing import Callable, Tuple, List, Optional
import numpy as np
from pseas.instance_selection.instance_selection import InstanceSelection
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


def __find_weights__(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    instances: int = x.shape[0]
    features: int = x.shape[1]

    qty: int = int(instances * (instances - 1) / 2)

    dx: np.ndarray = np.zeros((qty, features))
    dy: np.ndarray = np.zeros((qty,))

    # Compute dataset
    index: int = 0
    for i in range(instances):
        for j in range(i + 1, instances):
            dx[index] = x[i] - x[j]
            dy[index] = y[i] - y[j]
            index += 1
    np.square(dx, out=dx)
    np.abs(dy, out=dy)
    # np.square(dy, out=dy)

    # weights = argmin_w_i (norm [w_i (x_i -x'_i)]_i - |y - y'|)^2
    weights, residual = optimize.nnls(dx, dy)
    return np.sqrt(weights)

class UDD(InstanceSelection):
    """
    Uncertainty + alpha * Density - beta *  Diversity

    Uncertainty, Density, Diversity in [0;1] 

    k: int - number of neighbours for density step

    """

    def __init__(self, alpha: float = 1, beta: float = 1, k : int = 5) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.k : int = k

    def _dynamic_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.linalg.norm(self._weights * (x1 - x2))

    def ready(self, model, features, challenger_configuration, perf_matrix, **kwargs) -> None:
        self.model = model
        self.challenger_configuration = challenger_configuration
        self.n_instances: int = self.features.shape[0]
        # Find optimal distance function
        y: np.ndarray = np.median(perf_matrix, axis=1)
        self._weights: np.ndarray = __find_weights__(features, y)
        self._distances = __compute_distance_matrix__(features, self._dynamic_distance)

    def reset(self) -> None:
        """
        Reset this scheme so that it can start anew but with the same ready data.
        """
        pass

    def __uncertainty(self, not_done_instances) -> List[int]:
        """
        Original: Difference between max vote and max second vote for classification
        
        Ours: variance of predictions among forest"""
        uncertainty: np.ndarray = np.zeros(self.n_instances)
        for instance in not_done_instances:
            _, var = self.model.predict(self.challenger_configuration, instance)
            uncertainty[instance] = var
        return uncertainty

    def __k_nearest_neighbours(self, instance, not_done_instances):
        d = self._distances[instance, :]
        sorted = np.argsort(d)[::-1]
        k_best = []
        for i in sorted:
            if i in not_done_instances and i != instance:
                k_best.append(i)
                if len(k_best) == self.k:
                    break
        return k_best

    def __density(self, not_done_instances):
        densities = np.zeros(self.n_instances, float)
        for instance in not_done_instances[:]:
            neighbours = self.__k_nearest_neighbours(instance, not_done_instances)
            total: float = 0
            for neighbour in neighbours:
                dist: float = self._distances[instance, neighbour]
                total += dist*dist
            total /= len(neighbours)
            densities[instance] = total
        return densities


    def __diversity(self, not_done_instances):
        done_mask = np.array([i not in not_done_instances for i in range(self.n_instances)])
        diversities = np.min(self._distances[:, done_mask], axis=1)
        diversities[done_mask] = 0
        return diversities

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        not_done_instances: np.ndarray = np.array(
            [i for i, time in enumerate(state[0]) if time is None])
        
        uncertainties = self.__uncertainty(not_done_instances)
        # Normalize values in [0, 1]
        uncertainties -= np.min(uncertainties)
        uncertainties /= np.max(uncertainties)
        if self.alpha == 0 and self.beta == 0:
            scores = uncertainties
        else:
            densities = self.__density(not_done_instances)
            diversities = self.__diversity(not_done_instances)
            # Normalize values in [0, 1]
            densities -= np.min(densities)
            diversities -= np.min(diversities)
            densities /= np.max(densities)
            diversities /= np.max(diversities)
            scores = uncertainties + self.alpha * densities - self.beta * diversities
        for i in range(self.n_instances):
            if state[0][i] is not None:
                scores[i] = 1e30
        # How does that make sense?
        self._next = np.argmin(scores)


    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return "Uncertainty" if self.alpha == 0 and self.beta == 0 else "UDD"

    def clone(self) -> 'UDD':
        return UDD(self.alpha, self.beta, self.k)