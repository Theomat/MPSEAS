import csv
import os
from typing import Dict, Tuple
import numpy as np


def load_configuration_data(path: str) -> Tuple[Dict[str, int], Dict[str, int], np.ndarray]:
    """
    
    Return: (instance_name2int, configuration2int, data, features)
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
    
    return instance_name2int, configuration2int, data, features



if __name__ == "__main__":
    d = load_configuration_data("./rundata/kissat_ibm")
    assert d[2].shape == (684, 100), "Data matrix has incorrect size"
    assert np.min(d[2]) > 0, " A time should be positive"
    assert len(d[3]) == 684, "One feature vector per instance"
    print("All good!")
