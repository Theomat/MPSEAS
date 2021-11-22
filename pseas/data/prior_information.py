from typing import Any, List, Dict, Optional, Tuple

import numpy as np

import scipy.stats as st
import pseas.model as rf


def fill_features(features: Dict[int, np.ndarray], ninstances: int) -> np.ndarray:
    # Fill missing features with mean feature
    # Contains what's to fill
    to_fill: List[Tuple[int, Optional[np.np.ndarray]]] = []
    # Contains the sum of each feature that is not missing
    total_feature: np.ndarray = None
    # Contains the number of each feature that is not missing
    counts: np.ndarray = None

    for instance in range(ninstances):
        if instance not in features:
            to_fill.append((instance, None))
        else:
            feature = features[instance]
            missing: np.ndarray = np.isnan(feature)
            mask: np.ndarray = np.logical_not(missing)
            # Late initialisation to get the right array size
            if total_feature is None:
                total_feature = np.zeros_like(feature)
                counts = np.zeros_like(total_feature)
            total_feature[mask] += feature[mask]
            counts += mask
            if np.any(missing):
                to_fill.append((instance, missing))
    # Now total_feature will contain average feature
    total_feature /= counts
    # Fill missings
    for instance, mask in to_fill:
        if mask is None:
            features[instance] = total_feature.copy()
        else:
            (features[instance])[mask] = total_feature[mask]

    # To numpy array
    features_array = np.zeros((ninstances, total_feature.shape[0]))
    for i in range(ninstances):
        features_array[i] = features[i]

    return features_array


def initial_guess(distribution_name: str, data: np.ndarray) -> Dict[str, Any]:
    """
    Make an inital guess to parameters according to distribution and data.
    """
    if data.shape[0] == 0:
        return {}
    if distribution_name == "cauchy":
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        return {
            "loc": p50,
            "scale": (p75 - p25) / 2
        }
    elif distribution_name == "norm":
        return {
            "loc": np.mean(data),
            "scale": np.std(data)
        }
    return {}

def fit_same_class(distribution_name: str, perf_matrix: np.ndarray) -> np.ndarray:
    """
    Fit all the data of the perf matrix with instances of the same given distribution.
    """
    distribution = getattr(st, distribution_name)
    prior: np.ndarray = np.zeros(
        (perf_matrix.shape[0], 2), dtype=np.float64)
    for instance in range(perf_matrix.shape[0]):
        data = perf_matrix[instance, :]
        loc, scale = distribution.fit(data, **initial_guess(distribution_name, data))
        prior[instance, 0] = loc
        prior[instance, 1] = scale
    return prior

def fit_rf_model(features: np.ndarray, results: np.ndarray, configurations_dict: Dict[str, np.ndarray]) -> rf.Model:
    """
    Fit a random forest model on the data contained in results
    """
    model: rf.Model = rf.create_model()
    data = rf.create_dataset(features, configurations_dict, results)

    model.fit(data)
