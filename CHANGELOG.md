# GNNWR Changelog

## [Unreleased]

### Bug Fixes

- **修复 rescale() 逻辑错误**（datasets.py）：仅传入 x 或 y 之一时，因另一方 scale_info 为 None 导致误抛异常。改为独立检查 x 和 y。
- **修复除零错误**（datasets.py）：`minmax_scaler` 和 `standard_scaler` 在特征列所有值相同时（max == min 或 std == 0）触发除零。添加零值分母保护，常量列归一化后设为 0。
- **修复奇异矩阵崩溃**（utils.py）：DIAGNOSIS 中 X'X 矩阵奇异或病态时 `torch.linalg.inv` 崩溃。改用条件数检测 + 伪逆（`torch.linalg.pinv`）兜底。
- **修复 DataParallel 双重包装**（models.py）：模型保存后重新加载时 `torch.load` 返回的已经是 DataParallel 对象，代码会再次包装导致结构损坏。在 `run()`、`load_model()`、`reg_result()` 三处加入 `isinstance` 检查。
- **修复可变默认参数**（networks.py）：`SWNN`、`STPNN`、`STNN_SPNN` 的 `activate_func` 使用可变对象 `nn.PReLU()` / `nn.ReLU()` 作为默认参数，多个实例共享同一激活函数。改为 `None` + 运行时创建。
- **修复重复目录创建**（models.py）：`os.makedirs` 替换为 `os.makedirs(..., exist_ok=True)`。
- **修复变量名拼写错误**（models.py）：`weigth_decay` → `weight_decay`。

### Code Quality

- **替换废弃 API**（models.py）：将 `.data` 属性替换为 `.detach()`，遵循 PyTorch 官方推荐。
- **消除裸 except 子句**（models.py）：`except:` 改为具体异常类型，避免吞掉 `KeyboardInterrupt`。
- **修复 `__str__` 方法**（models.py）：原实现通过 `print()` 输出并返回空字符串，改为返回格式化字符串。
- **修复 LinearNetwork `__str__`**（networks.py）：处理 `drop_out=0` 时 `nn.Identity` 没有 `.p` 属性的情况。

### Performance

- **ManhattanDistance 优化**（datasets.py）：将 numpy 广播运算替换为 `scipy.spatial.distance.cdist`，加速 6-11x。
- **BasicDistance 多后端支持**（datasets.py）：新增 `backend`（'scipy'/'torch'/'auto'）和 `device` 参数，支持 GPU 加速距离计算。默认行为不变。
- **DataLoader 张量缓存**（datasets.py）：`baseDataset.__getitem__` 使用预转换张量缓存，避免每次调用 `torch.tensor()` 重复分配内存，加速 3-7x。

### New Features

- **KNN 稀疏距离矩阵**（datasets.py, models.py）：新增 `KNNDistance()` 函数，只存储 K 个最近邻的距离，大幅降低内存占用。`init_dataset` 新增 `knn_k`、`reference_size` 参数。10K 数据从 305MB 降至 7.6MB（97.5%），100K 数据从 29.8GB 降至 76MB（99.8%）。
- **DIAGNOSIS lite 模式**（utils.py, models.py）：大数据集（n > 10000）自动跳过 Hat 矩阵计算（O(n²) 内存和 O(n³) 计算），R²、RMSE 等核心指标仍可正常使用。
