# Autogluon onnx Generator 
**该库包含目前已经实现的autogluon bagging模型的onnx转化器**

# Introduction 

**目前包含了Fastai,LightGBM, Random Forest, Weighted_Ensemble，XGboost的集成模型转onnx的代码。**

# Usage 

***TODO: 调用API还没有完成***

调用方法：

    1. 打开需要转换模型的对应脚本，比如：LightGBM_Bagging_Generator.py
    2. 定义模型路径参数， 比如：model_dir = r'E:\Bagging2onnx\Autogluon2onnx\autogluon_USRS_first_cls\models\LightGBMLarge_BAG_L1'
    3. 实例化类 model = LightGBM_bagging_onnx_generator(model_dir=model_dir)
    4. 调用transform方法model.transform()生成onnx文件
