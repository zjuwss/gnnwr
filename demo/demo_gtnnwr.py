import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from src.gnnwr.datasets import init_dataset, init_dataset_cv
from src.gnnwr.models import GTNNWR

data = pd.read_csv(r'../data/simulated_data_GTNNWR.csv')
data["id"] = np.arange(len(data))
train_dataset, val_dataset, test_dataset = init_dataset(data=data,
                                                        test_ratio=0.15,
                                                        valid_ratio=0.15,
                                                        x_column=['x1', 'x2'],
                                                        y_column=['Z'],
                                                        spatial_column=['u', 'v'],
                                                        temp_column=['t'],
                                                        id_column=['id'],
                                                        sample_seed=10,
                                                        batch_size=128)
SGD_PARAMS = {
    "maxlr": .1,
    "minlr": 0.002,
    "upepoch": 5000,
    "decayepoch": 15000,
    "decayrate": 0.999,
}
gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset, [[3], [1024, 512, 256]], optimizer='SGD',
                optimizer_params=SGD_PARAMS)
# gtnnwr.add_graph()
gtnnwr.run(10000)
# gtnnwr.reg_result('../result/gtnnwr_result.csv')
