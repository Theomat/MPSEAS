from typing import Tuple
import pyrfr


class Model():
    def __init__(self, forest, rng) -> None:
        self.forest = forest
        self.rng = rng

    def predict(self, configuration, instance) -> Tuple[float, float]:
        """
        Return result and incertitude
        """
        # TODO
        self.forest.predict(configuration, instance)
        pass


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
