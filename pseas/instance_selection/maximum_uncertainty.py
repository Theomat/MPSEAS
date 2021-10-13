from typing import Tuple, List, Optional
import numpy as np
from pseas.instance_selection.instance_selection import InstanceSelection


class MaximumUncertainty(InstanceSelection):
    """
    Instance selection method interface.
    Basically method calls to a strategy call the same method if it exists in the instance selection method.
    """

    def __init__(self, samples : int = 100) -> None:
        super().__init__()
        self.samples: int = samples

    def ready(self, model, configuration_distribution, **kwargs) -> None:
        self.model = model
        self.configuration_distribution = configuration_distribution

    def reset(self) -> None:
        """
        Reset this scheme so that it can start anew but with the same ready data.
        """
        pass

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        not_done_instances: np.ndarray = np.array(
            [i for i, time in enumerate(state[0]) if time is None])
        
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

        self._next = np.argmax(uncertainty)


    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return "maximum uncertainty"

    def clone(self) -> 'MaximumUncertainty':
        return MaximumUncertainty(self.samples)