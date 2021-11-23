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
            instance_features[instance_name2int[row[0]]] = np.array([np.double(val) for val in row[1:]],dtype=np.double)

    # convert configuration strings into lists of double
    with open(os.path.join(path, "params.pcs")) as pcs_file:
        pcs_list = pcs_file.readlines()
        #breakpoint()
        parameter_space = read_pcs(pcs_list)
    #print(parameter_space.get_default_configuration().get_array())
    #TODO: this is a bit hacky, will work only for kissat for now
    #configurations: Dict[int, np.ndarray]={}
    #parameters=set()
    #for conf in configuration2int.keys():
    #    for param in conf.lstrip('-').split(' -'):
    #        parameters.add(param.split(" '")[0])
    #parameter_list=np.array(sorted(list(parameters)))

    #default=np.full(len(parameter_list),None)
    #with open(os.path.join(path, "default.txt")) as fd:
    #    row=fd.readline()
    #    for param in row.lstrip('-').split(' -'):
    #        
    #        param_name, param_value = param.strip("'").split(" '")
    #        default[np.where(parameter_list == param_name)]=param_value

    # for conf in configuration2int.keys():
    #     configurations[configuration2int[conf]]=default
    #     for param in conf.lstrip('-').split(' -'):
    #         param_name, param_value = param.strip("'").split(" '")
    #         param_value= 0 if param_value=='false' else 1 if param_value=='true' else param_value
    #         try:
    #             param_value=float(param_value)
    #         except:
    #             pass
    #         configurations[configuration2int[conf]][np.where(parameter_list == param_name)] = param_value
    #     configurations[configuration2int[conf]]=configurations[configuration2int[conf]].astype(np.double)
    configurations: Dict[int, np.ndarray]={}
    for conf in configuration2int.keys():
        config_dict: Dict[str,str] = {}
        for param in conf.lstrip('-').split(' -'):
            param_name, param_value = param.strip("'").split(" '")
            config_dict[param_name] = param_value
        ConfigSpace.util.fix_types(config_dict,parameter_space)
        config_object = ConfigSpace.Configuration(parameter_space,config_dict)
        configurations[configuration2int[conf]]=config_object.get_array()
        #print(configurations[configuration2int[conf]])
            
    return data, instance_features, configurations



if __name__ == "__main__":
    d = load_configuration_data("./rundata/kissat_ibm")
    #print(d[2][1])
    assert d[0].shape == (684, 100), "Data matrix has incorrect size"
    assert np.min(d[0]) > 0, " A time should be positive"
    assert len(d[1]) == 684, "One feature vector per instance"
    print("All good!")
