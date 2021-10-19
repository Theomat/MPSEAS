import csv
import os
from typing import Dict, Tuple
import numpy as np


def load_configuration_data(path: str) -> Tuple[Dict[str, int], Dict[str, int], np.ndarray]:
    """
    
    Return: (instance_name2int, configuration2int, data)
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

    return instance_name2int, configuration2int, data



if __name__ == "__main__":
    d = load_configuration_data("./rundata/kissat_ibm")
    assert d[2].shape == (684, 100)
    print("All good!")