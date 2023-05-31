import os
import pickle

import fire
from autogluon.core.models.ensemble.bagged_ensemble_model import \
    BaggedEnsembleModel
from autogluon.core.models.ensemble.weighted_ensemble_model import \
    WeightedEnsembleModel
from autogluon.tabular.models import *

from onnx_generators import *


def main(model_path):
    with open(os.path.join(model_path,'model.pkl'),'rb') as f:
        model = pickle.load(f)
    if isinstance(model, WeightedEnsembleModel):
        model = Weighted_Ensemble_onnx_generator(model_path)
    elif isinstance(model, BaggedEnsembleModel):
        child_type = model._child_type
        if child_type == RFModel:
            model = RandomForest_Bagging_onnx_generator(model_path)
        elif child_type == XGBoostModel:
            model = XGboost_Bagging_onnx_generator(model_path)
        elif child_type ==  LGBModel:
            model = LightGBM_Bagging_onnx_Generator(model_path)
        elif child_type == NNFastAiTabularModel:
            model = Fastai_Bagging_onnx_Generator(model_path)
        elif child_type == TabularNeuralNetTorchModel:
            model = NeuralTorch_Bagging_onnx_generator(model_path)
        else:
            raise ValueError('model type not supported')
    else:
        raise ValueError('model type not supported')
    model.transform()
    print('The onnx model is saved in {}'.format(os.path.join(os.path.dirname(model.bagged_model_path),'onnx_model.onnx')))

if __name__ == '__main__':
    fire.Fire(main)
    # main(r'E:\Bagging2onnx\AutogluonOnnxGenerator_1.0\autogluon_USCPI_first_cls\models\LightGBMLarge_BAG_L1')
