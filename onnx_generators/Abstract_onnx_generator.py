
import argparse
import os
import pickle
from os.path import isdir, isfile, join

import lightgbm as lgb
import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabular_nn.torch.torch_network_modules import \
    EmbedNet
from lightgbm.basic import Booster as lgbmBooster
from onnxconverter_common.data_types import (DictionaryType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from skl2onnx import convert_sklearn

from .operators import argmax_operator, mean_operator, softmax_operator
from .utils import model_dir_tools


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",type=str)
    args = parser.parse_args()
    return args


class Abstract_ONNX_Generator():

    """
    This class is used to load the bagged model and convert it to onnx format.
    Warning: The outputs of LightGBM onnx model are not always as same as the original model.

    Parameters
    ----------
    model_path : str
        Path to the bagged_model in autogulon directory. e.g. 'AutogluonModels/ag_model_2021-05-20_15-00-00/models/LightGBMLarge'
    """
    def __init__(self,model_dir) -> None:

        self.bagged_model_path, self.children_path = model_dir_tools(model_dir)
        self.bagged_model = self.load_model(self.bagged_model_path)
        self.problem_type = self.bagged_model.problem_type
        self.children = [self.bagged_model.load_child(child) for child in self.bagged_model.models]
        self.input_format = list(self.bagged_model.feature_metadata.get_type_group_map_raw().items())
    
    
    @property
    def load_torch_onnx(self):
        self.torch_onnx = [onnx.load(os.path.join(child.path,'torch_model.onnx')) for child in self.children]


    def load_model(self,model_path):
        """
        This function is used to load models.
        """
        if isinstance(model_path, str):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif isinstance(model_path, list):
            model_list = []
            for dir in list:
                with open(dir, 'rb') as f:
                    model_list.append(pickle.load(f)) 
            return model_list
    

    def load_onnx_models(self,children_path):
        """
        This function is used to load the onnx models.
        """
        if children_path:
            if isinstance(children_path[0],str):
                onnx_models = [onnx.load(join(os.path.dirname(children_path[i]),'onnx_model.onnx')) for i in range(len(children_path))]
        return onnx_models
    
    def rename_node_name(self,onnx_models):
        """
        This function is used to rename the node name of onnx models.
        """
        for i in range(len(onnx_models)):
            graph = onnx_models[i].graph
            for initializer in graph.initializer:
                initializer.name = initializer.name + str(i)
            for z in range(len(graph.output)):
                graph.output[z].name = graph.output[z].name + str(i)
            nodes = graph.node
            for j in range(len(nodes)):
                nodes[j].name = nodes[j].name + str(i)
                if j == 0:
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
                else:
                    for k in range(len(nodes[j].input)):
                        nodes[j].input[k] = nodes[j].input[k] + str(i)
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
        return onnx_models
   

    def merge_graphs(self,merged_model,onnx_models):
        """
        This function is used to merge the onnx models.
        """
        if isinstance(onnx_models,list):
            # if isinstance(self.children[0].model,EmbedNet):
            #     for onnx_model in self.children:
            #         graph = onnx_model.graph
            #         graph.node.extend(self.operators['softmax'])
            # self.merged_onnx_model = self.onnx_models[0]
            # self.graph = self.merged_onnx_model.graph
            # for i in range(1,len(self.onnx_models)):    
            #     for node in self.onnx_models[i].graph.node:
            #         self.graph.node.extend([node])
            #     for initializer in self.onnx_models[i].graph.initializer:
            #         self.graph.initializer.extend([initializer])
            merged_model.graph.input.extend([input for input in onnx_models[0].graph.input])
            for onnx_model in onnx_models:
                merged_model.graph.initializer.extend([initializer for initializer in onnx_model.graph.initializer])
                merged_model.graph.node.extend([node for node in onnx_model.graph.node])
            return merged_model
        
        else:
            raise NotImplementedError
        

    
    def save(self,onnx_model):
        with open (join(os.path.dirname(self.bagged_model_path),'onnx_model.onnx'),'wb') as f:
            f.write(onnx_model.SerializeToString())


# if __name__ == "__main__":
#     # args = get_args()
#     # bagged_model = Abstract_ONNX_Generator(args.model_dir)
#     # bagged_model.transform()
#     bagged_model = Abstract_ONNX_Generator(args.model_dir)