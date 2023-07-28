import os
import sys
import pandas as pd

sys.path.append(os.getcwd())
from src.datasets import init_dataset, init_dataset_cv
from src.models import GTNNWR

data = pd.read_csv(r'../data/demo_data_gtnnwr.csv')

train_dataset, val_dataset, test_dataset = init_dataset(data=data,
                                                        test_ratio=0.2,
                                                        valid_ratio=0.1,
                                                        x_column=['refl_b01','refl_b02','refl_b03','refl_b04','refl_b05'],
                                                        y_column=['SiO3'],
                                                        spatial_column=['proj_x', 'proj_y'],
                                                        temp_column=['day'],
                                                        sample_seed=42,)

gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset)

gtnnwr.run(100)
