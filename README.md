# gnnwr

<p>
    <img title="python version" src="https://img.shields.io/badge/python-3.9-blue" alt="">
    <!--<img title="PyPI" src="https://img.shields.io/pypi/v/:packageName"  alt="PyPI">-->
</p>


A pytorch implementation of the Geographically Neural Network Weighted Regression (GNNWR) and its derived models

This repository contains:

1. Source code of GNNWR, GTNNWR model and other derived models
2. Tutorial notebooks of how to use these model
3. ...

## Table of Contents

- [Models](#models)
- [Install](#install)
- [Usage](#usage)
- [Reference](#reference)
- [Contributing](#contributing)
- [License](#license)

## Models

- [GNNWR](https://doi.org/10.1080/13658816.2019.1707834): Geographically neural network weighted regression, a model address spatial non-stationarity in various domains with complex geographical processes. A spatially weighted neural network (SWNN) is proposed to represent the nonstationary weight matrix.
- [GTNNWR](https://doi.org/10.1080/13658816.2020.1775836): Geographically and temporally neural network weighted regression, a model for estimating spatiotemporal non-stationary relationships. A spatiotemporal proximity neural network (STPNN) is proposed in this paper to accurately generate space-time distance.

## Install

Using pip to install gnnwr:  

```
pip install gnnwr
```

## Usage

We provide a series of encapsulated methods and predefined default parameters, users only need to use to load dataset with `pandas` , and call the functions in `gnnwr` package to complete the regression:

```python
from gnnwr import models,datasets
import pandas as pd

data = pd.read_csv('your_data.csv')

train_dataset, val_dataset, test_dataset = datasets.init_dataset(data=data,
                                                        		 test_ratio=0.2, valid_ratio=0.1,
                                                                 x_column=['x1', 'x2'], y_column=['y'],
                                                        		 spatial_column=['u', 'v'])

gnnwr = models.GNNWR(train_dataset, val_dataset, test_dataset)

gnnwr.run(100)
```

For other uses of customization, the [demos](https://github.com/zjuwss/gnnwr/tree/main/demo) can be referred to.

## Reference

### algorithm  

1. Du, Z., Wang, Z., Wu, S., Zhang, F., and Liu, R., 2020. Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity. International Journal of Geographical Information Science, 34 (7), 1353–1377.  
2. Wu, S., Wang, Z., Du, Z., Huang, B., Zhang, F., and Liu, R., 2021. Geographically and temporally neural network weighted regression for modeling spatiotemporal non-stationary relationships. International Journal of Geographical Information Science, 35 (3), 582–608.


### case study demo



## Contributing

### Contributors



## License
[GPLv3 license](https://github.com/zjuwss/gnnwr/blob/main/LICENSE)


