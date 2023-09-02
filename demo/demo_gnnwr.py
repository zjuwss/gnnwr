import os
import sys

import pandas as pd
sys.path.append(os.getcwd())
from src.gnnwr.datasets import init_dataset,init_dataset_cv,init_predict_dataset
from src.gnnwr.models import GTNNWR,GNNWR

#23.6.8_TODO: demo需要包括：cv与非cv版本、gnnwr与gtnnwr、规范注释（英文）、

data = pd.read_csv(u'..\\data\\simulated_data_GTNNWR.csv')
# distances = pd.read_csv(u'..\\data\\distances.csv')
# predict_data = pd.read_csv(u'..\\data\\demo_predict_data.csv')
# k-fold 导入数据
# cv_dataset, test_dataset = init_dataset_cv(
#     data, 0.2, 5, ['x1', 'x2'], ['y'], ['u', 'v'], ['id'])

# gnnwr train test
# train_dataset, val_dataset, test_dataset = init_dataset(
#     data, 0.2, 0.1, ['x1', 'x2'], ['y'],['u','v'])

# gtnnwr train test
train_dataset, val_dataset, test_dataset = init_dataset(data=data,
                                                        test_ratio=0.2,
                                                        valid_ratio=0.1,
                                                        x_column=['x1', 'x2'],
                                                        y_column=['Z'],
                                                        spatial_column=['u', 'v'],
                                                        id_column=['id'],
                                                        sample_seed=42)
# predict_dataset = init_predict_dataset(predict_data,train_dataset,x_column=['refl_b01','refl_b02','refl_b03','refl_b04','refl_b05'],spatial_column=['proj_x', 'proj_y'],temp_column=['day'])
# print(predict_dataset.x_data)
# 使用计算好的距离
# train_dataset, val_dataset, test_dataset = init_dataset_with_dist_frame(data, 0.7, 0.1, ['x1', 'x2'], ['y'], ['id'], distances)

# GNNWR Test
SGD_PARAMS = {
    "maxlr": 0.3,
    "minlr": 0.1,
    "upepoch": 5000,
    "decayepoch": 15000,
    "decayrate": 0.999,
}
gnnwr = GNNWR(train_dataset, val_dataset, test_dataset, [128,64,32, 8],optimizer="SGD",optimizer_params=SGD_PARAMS)


# gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset, [[3],[512,256, 128, 64,32,16,8]],optimizer="SGD",optimizer_params=SGD_PARAMS)

# gtnnwr.load_model("../gtnnwr_models/GNNWR_230609.pkl")
# print("start running!")
gnnwr.run(50000)
gnnwr.add_graph()