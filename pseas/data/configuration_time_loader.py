import csv
import os
from typing import Dict, Tuple
import numpy as np


def load_configuration_data(path: str) -> Tuple[Dict[str, int], Dict[str, int], np.ndarray]:
    """
    
    Return: (instance_name2int, configuration2int, data, features, configurations)
    """
    instance_name2int: Dict[str, int] = {}
    configuration2int: Dict[str, int] = {}
    with open(os.path.join(path, "configlist.csv")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            configuration2int[row[1]] = int(row[0])

    with open(os.path.join(path, "instancelist.csv")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            instance_name2int[row[1]] = int(row[0])

    data = np.zeros((len(instance_name2int), len(configuration2int)), dtype=float)
    with open(os.path.join(path, "perflist.csv")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            data[int(row[1]), int(row[0])] = float(row[2])

    features: Dict[int, list]={}
    with open(os.path.join(path, "features.txt")) as fd:
        reader = csv.reader(fd)
        # Skip first line
        next(reader)
        for row in reader:
            features[instance_name2int[row[0]]] = [float(val) for val in row[1:]]

    configurations: Dict[str, np.ndarray]={}
    parameters=set()
    for conf in configuration2int.keys(): 
        for param in conf.lstrip('-').split(' -'):
            parameters.add(param.split(" '")[0])
    parameter_list=np.array(sorted(list(parameters)))

    default=np.full(len(parameter_list),None)
    with open(os.path.join(path, "default.txt")) as fd:
        row=fd.readline()
        for param in row.lstrip('-').split(' -'):
            
            param_name, param_value = param.strip("'").split(" '")
            default[np.where(parameter_list == param_name)]=param_value

    for conf in configuration2int.keys():
        configurations[str(configuration2int[conf])]=default
        for param in conf.lstrip('-').split(' -'):
            param_name, param_value = param.strip("'").split(" '")
            try:
                param_value=float(param_value)
            except:
                pass
            configurations[configuration2int[conf]][np.where(parameter_list == param_name)] = param_value
            
    return instance_name2int, configuration2int, data, features, configurations



if __name__ == "__main__":
    d = load_configuration_data("./rundata/kissat_ibm")
    print(d[4])
    assert d[2].shape == (684, 100), "Data matrix has incorrect size"
    assert np.min(d[2]) > 0, " A time should be positive"
    assert len(d[3]) == 684, "One feature vector per instance"
    print("All good!")
