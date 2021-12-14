import csv
import os
from typing import Dict, Tuple
import numpy as np
from ConfigSpace.read_and_write.pcs import read as read_pcs
import ConfigSpace
import ConfigSpace.util


def load_configuration_data(path: str) -> Tuple[np.ndarray, Dict[int,np.ndarray], Dict[int,np.ndarray]]:
    """
    reads the files in the given directory and returns the performances, instances and configurations
    The directory needs to contain the following files:
        configlist.csv : contains configuration_id, configuration_string
        instancelist.csv : contain instance_id, instance_name
        perflist.csv : contains configuration_id, instance)id, performance
        features.txt : contains instance_name, one column per feature
        default.txt : contains the string of the default configuration
    Return: (data, instance_features, configurations)
    """

    #TODO : we shoult probably use the configuration space description file and the ConfigSpace library
    #load configuration strings
    instance_name2int: Dict[str, int] = {}
    configuration2int: Dict[str, int] = {}
    with open(os.path.join(path, "configlist.csv")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            configuration2int[row[1]] = int(row[0])

    # load instance names and id
    with open(os.path.join(path, "instancelist.csv")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            instance_name2int[row[1]] = int(row[0])

    # load performance data
    data = np.zeros((len(instance_name2int), len(configuration2int)), dtype=np.double)
    with open(os.path.join(path, "perflist.csv")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            data[int(row[1]), int(row[0])] = np.double(row[2])

    # load instances features
    instance_features: Dict[int, np.array]={}
    with open(os.path.join(path, "features.txt")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            if row[0] in instance_name2int.keys():
                instance_features[instance_name2int[row[0]]] = np.array([np.double(val) for val in row[1:]],dtype=np.double)
        ## This should sanitize the instance features and take out NaN values (not tested)
        col_mean = [np.mean([instance_features[key][ind]
                             for key in instance_features.keys()
                             if instance_features[key][ind] != np.NaN
                             ])
                    for ind in range(len(instance_features[0]))]
        for key in instance_features.keys():
            instance_features[key] = np.array([instance_features[key][ind]
                                               if instance_features[key][ind] != np.NaN
                                               else col_mean[ind]
                                               for ind in range(len(instance_features[0]))], dtype=np.double)
    
    # convert configuration strings into lists of double
    with open(os.path.join(path, "params.pcs")) as pcs_file:
        pcs_list = pcs_file.readlines()
        #breakpoint()
        parameter_space = read_pcs(pcs_list)
    configurations: Dict[int, np.ndarray] = {}
    #get list of default values
    default_config = parameter_space.get_default_configuration().get_array()
    for key in parameter_space._hyperparameters:
        hyper_parameter = parameter_space.get_hyperparameter(key)
        default_config[parameter_space._hyperparameter_idx[key]] = hyper_parameter._inverse_transform(hyper_parameter.default_value)
    for conf in configuration2int.keys():
        config_dict: Dict[str,str] = {}
        for param in conf.lstrip('-').split(' -'):
            param_name, param_value = param.strip("'").split(" '")
            config_dict[param_name] = param_value
        ConfigSpace.util.fix_types(config_dict, parameter_space)
        config_object = ConfigSpace.Configuration(parameter_space, config_dict)
        configurations[configuration2int[conf]] = config_object.get_array()
        #replace nan values (non active parameters) by their default value
        configurations[configuration2int[conf]] = np.array([configurations[configuration2int[conf]][ind]
                                                            if configurations[configuration2int[conf]][ind] == configurations[configuration2int[conf]][ind]
                                                            else default_config[ind]
                                                            for ind in range(len(configurations[configuration2int[conf]]))])
        #print(default_config)
            
    return data, instance_features, configurations



if __name__ == "__main__":
    d = load_configuration_data("./rundata/cplex_regions200")
    #print(d[1][2])
    assert d[0].shape == (684, 100), "Data matrix has incorrect size"
    assert np.min(d[0]) > 0, " A time should be positive"
    assert len(d[1]) == 684, "One feature vector per instance"
    print("All good!")
