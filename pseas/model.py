from typing import Dict, Tuple
import pyrfr.regression
import numpy as np


class Model():
    def __init__(self, forest, conf, features, rng) -> None:
        self.forest = forest
        self.rng = rng
        self.conf = conf
        self.features = features

    def predict(self, configuration, instance) -> Tuple[float, float]:
        """
        Return result and incertitude
        """
        feature_vector = self.conf[configuration].tolist() + self.features[instance].tolist()
        return self.forest.predict_mean_var(feature_vector)

    def fit(self, data: pyrfr.regression.default_data_container_with_instances):
        # TODO: Change to something relevant, this was put just ot have an on zero number (otherwise it crashes)
        self.forest.options.num_data_points_per_tree = data.num_data_points() # means same number as data points

        self.forest.fit(data, self.rng)


def create_model(conf, features, num_trees: int = 10, seed: int = 0) -> Model:
    #reset to reseed the rng for the next fit
    rng = pyrfr.regression.default_random_engine(seed)
    # create an instance of a regerssion forest using binary splits and the RSS loss
    the_forest = pyrfr.regression.binary_rss_forest()

    the_forest.options.num_trees = num_trees
    # the forest's parameters
    the_forest.options.do_bootstrapping = True              # default: false
    the_forest.options.tree_opts.min_samples_to_split = 3   # 0 means split until pure
    the_forest.options.tree_opts.min_samples_in_leaf = 3    # 0 means no restriction
    the_forest.options.tree_opts.max_depth = 2048           # 0 means no restriction
    the_forest.options.tree_opts.epsilon_purity = 1e-8	    # when checking for purity, the data points can differ by this epsilon


    return Model(the_forest, conf, features, rng)

def create_dataset(instance_features: np.ndarray, configurations: Dict[int, np.ndarray], data : np.ndarray) -> pyrfr.regression.default_data_container_with_instances:
    conf_len: int = len(configurations[list(configurations.keys())[0]])
    feat_len: int = instance_features.shape[1]
    forest_data : pyrfr.regression.default_data_container_with_instances = pyrfr.regression.default_data_container_with_instances(conf_len, feat_len)
    for c in configurations.keys():
        forest_data.add_configuration(list(configurations[c]))
    for inst in range(instance_features.shape[0]):
        forest_data.add_instance(instance_features[inst,:])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not data[i,j] == np.nan:
                forest_data.add_data_point(j, i, data[i,j])
    return forest_data

