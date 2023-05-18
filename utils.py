import os
from os.path import isdir, isfile, join

from onnxconverter_common.data_types import (DictionaryType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType,
                                             StringTensorType)
from torch import nn


def model_dir_tools(model_dir):
    """
    This function is used to get the model path and child model path.
    Parameters
    ----------
    model_dir : str
        Path to the bagged_model in autogulon directory.
    Returns
    -------
    bagged_model_path : str
        Path to the bagged_model.pkl.
    children_dir : list
        List of child model path.
    """
    
    bagged_model_path = join(model_dir,'model.pkl')
    children_dir = os.listdir(model_dir)
    children_new_dir = []
    for child_dir in children_dir:
        if child_dir != 'utils':
            if isdir(join(model_dir,child_dir)):
                child_dir = join(model_dir,child_dir,'model.pkl')
                children_new_dir.append(child_dir)
    return bagged_model_path,children_new_dir


class NeuralTorchModel(nn.Module):
    def __init__(self,main_block,softmax):
        super().__init__()
        self.main_block = main_block
        self.softmax = softmax

    def forward(self, x):
        return self.softmax(self.main_block(x))

class FastAIModel(nn.Module):
    def __init__(self,bn_cont,layers,softmax):
        super().__init__()
        self.bn_cont = bn_cont
        self.layers = layers
        self.softmax = softmax

    def forward(self, x):
        return self.softmax(self.layers(self.bn_cont(x)))

def convert_dataframe_schema(input_format, drop=None):
    inputs = []
    for k, v in input_format:
        if drop is not None and k in drop:
            continue
        if k == 'float':
            inputs.extend([(v[i], FloatTensorType([1, 1])) for i in range(len(v))])
        elif k == 'int':
            inputs.extend([(v[i], Int64TensorType([1, 1])) for i in range(len(v))])
        elif k == 'string':
            inputs.extend([(v[i], StringTensorType([1, 1])) for i in range(len(v))])
        else:
            raise NotImplementedError
    return inputs