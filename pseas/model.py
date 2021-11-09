from typing import Dict, Tuple
import pyrfr.regression
import numpy as np


class Model():
    def __init__(self, forest, rng) -> None:
        self.forest = forest
        self.rng = rng

    def predict(self, configuration, instance) -> Tuple[float, float]:
        """
        Return result and incertitude
        """
        # TODO
        feature_vector = configuration + instance
        return self.forest.predict_mean_var(feature_vector)

    def fit(self, data):
        self.forest.fit(data, self.rng)


def create_model(num_trees: int = 10, seed: int = 0) -> Model:
    #reset to reseed the rng for the next fit
    rng = pyrfr.regression.default_random_engine(seed)
    # create an instance of a regerssion forest using binary splits and the RSS loss
    the_forest = pyrfr.regression.binary_rss_forest()

    the_forest.options.num_trees = num_trees
    # the forest's parameters
    the_forest.options.do_bootstrapping = True	# default: false
    the_forest.options.tree_opts.min_samples_to_split = 3	# 0 means split until pure
    the_forest.options.tree_opts.min_samples_in_leaf = 3	# 0 means no restriction 
    the_forest.options.tree_opts.max_depth = 2048			# 0 means no restriction
    the_forest.options.tree_opts.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon

    return Model(the_forest, rng)

def create_dataset(instance_features: Dict[int, np.ndarray], configurations: Dict[int, np.ndarray], data : np.ndarray) -> pyrfr.regression.data_base:
    conf_len: int = len(configurations[list(configurations.keys())[0]])
    feat_len: int = len(instance_features[list(instance_features.keys())[0]])
    forest_data = pyrfr.regression.default_data_container_with_instances(conf_len, feat_len)
    for c in configurations.keys():
        forest_data.add_configuration(list(configurations[c]))
    for inst in instance_features.keys():
        forest_data.add_instance(list(instance_features[inst]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            forest_data.add_data_point(j, i, data[i][j])
    return data

