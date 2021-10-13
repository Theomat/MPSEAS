from typing import Callable, Tuple, List, Optional
import numpy as np
from pseas.instance_selection.instance_selection import InstanceSelection

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


class UDD(InstanceSelection):
    """
    Uncertainty + alpha * Density + beta *  Diversity

    Uncertainty, Density, Diversity in [0;1] 

    samples: int - number of configuration samples to take for uncertainty step
    k: int - number of neighbours for density step

    TODO: missing self.distance
    """

    def __init__(self, samples : int = 100, alpha: float = 1, beta: float = 1, k : int = 5) -> None:
        super().__init__()
        self.samples: int = samples
        self.alpha: float = alpha
        self.beta: float = beta
        self.k : int = 5

    def ready(self, model, configuration_distribution, features, **kwargs) -> None:
        self.model = model
        self.configuration_distribution = configuration_distribution
        self._distances = __compute_distance_matrix__(features, self.distance)
        self.n_instances: int = self.features.shape[0]

    def reset(self) -> None:
        """
        Reset this scheme so that it can start anew but with the same ready data.
        """
        pass

    def __uncertainty(self, not_done_instances) -> List[int]:
        """
        Original: Difference between max vote and max second vote for classification
        
        Ours: |E[model_uncertainty(instance, X)] where X ~ configuration"""
        uncertainty: np.ndarray = np.zeros(self.n_instances)
        for instance in not_done_instances:
            configurations = self.configuration_distribution(self.samples)
            total_uncertainty: float = 0
            coeff_sums: float = 0
            for p, configuration in configurations:
                _, uncert = self.model(configuration, instance)
                coeff_sums += p
                total_uncertainty += uncert * p
            uncertainty[instance] = total_uncertainty / max(1e-10, coeff_sums)
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
        densities = self.__density(not_done_instances)
        diversities = self.__diversity(not_done_instances)
        # Normalize values in [0, 1]
        uncertainties -= np.min(uncertainties)
        densities -= np.min(densities)
        diversities -= np.min(diversities)
        uncertainties /= np.max(uncertainties)
        densities /= np.max(densities)
        diversities /= np.max(diversities)

        scores = uncertainties + self.alpha * densities + self.beta * diversities

        self._next = np.argmax(scores)


    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return "UDD"

    def clone(self) -> 'UDD':
        return UDD(self.samples, self.alpha, self.beta, self.k)