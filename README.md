# CIFAR-10 图像分类实验

本实验在 CIFAR-10 上训练卷积神经网络，固定模型结构和数据划分，比较数据增强与余弦学习率调度对分类结果的影响。两种方法合用时，第 21 轮的验证准确率最高，为 **87.50%**。采用这一轮的权重评估官方测试集，10,000 张图像中有 **8,642 张分类正确，总准确率为 86.42%**。

## 直接复核测试成绩

评估脚本默认加载仓库中的[第 21 轮权重](runs/augment_cosine_seed42/best.pt)和对应的[训练配置](runs/augment_cosine_seed42/config.json)。实验环境为 Python 3.12.10、PyTorch 2.6.0、CUDA 12.4 和 RTX 4060 Laptop GPU。在 Windows PowerShell 中运行以下命令即可复核成绩。

```powershell
git clone https://github.com/sakannnna/cifar10-cnn-classification.git
cd cifar10-cnn-classification
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe evaluate_test.py --download --out verification/test
```

首次运行时，脚本会下载 CIFAR-10 并校验文件。已有官方压缩包时，可将 `cifar-10-python.tar.gz` 放入 `data/`，脚本会校验并解压。使用 CPU 评估时，在最后一条命令末尾加上 `--device cpu`。

从干净目录复核时，程序输出 `Epoch 21 | correct 8642/10000 | accuracy 86.42%`。`verification/test/predictions.csv` 中的 10,000 条预测与仓库保存的[原始记录](results/seed42/test/predictions.csv)逐条一致。

## 四组训练结果

CIFAR-10 包含十类 32×32 彩色图像。官方训练集每类有 5,000 张，按类别划分后，训练集共 45,000 张，验证集共 5,000 张。四组共用同一份划分索引和种子 42，各从头训练 25 轮。

网络由三组卷积提取特征，通道数依次为 32、64、128，最后通过全局平均池化和线性层给出十个类别的分数。各组均使用 Adam 优化器和交叉熵损失，批大小为 128，按验证准确率选择权重。

| 训练方法 | 最佳验证准确率 | 最佳轮次 | 第 25 轮验证准确率 | 训练记录 |
| --- | ---: | ---: | ---: | --- |
| 基础 CNN | 83.48% | 25 | 83.48% | [CSV](runs/baseline_seed42/history.csv) |
| 仅余弦调度 | 85.38% | 24 | 85.22% | [CSV](runs/cosine_seed42/history.csv) |
| 仅数据增强 | 86.06% | 25 | 86.06% | [CSV](runs/augment_seed42/history.csv) |
| 数据增强＋余弦调度 | **87.50%** | **21** | 87.38% | [CSV](runs/augment_cosine_seed42/history.csv) |

增强组在训练时给图像四周各补 4 个像素的黑边，随机裁回原尺寸，再以 50% 的概率水平翻转。固定学习率组全程使用 0.001，余弦调度组在 25 轮内从 0.001 降至 0.00001。验证时四组都使用原图，将像素值缩放到 [−1, 1]。

按最佳验证准确率比较，单用调度比基础组高 1.90 个百分点，单用增强高 2.58 个百分点。在已有增强的基础上加入调度，验证集又多分对了 72 张，准确率从 86.06% 升至 87.50%。

![基础组和增强加调度组的训练与验证曲线](results/seed42/learning_curves.png)

## 测试结果与典型错例

测试集每类各有 1,000 张图像。汽车的准确率最高（94.7%），猫最低（69.4%）。120 张猫被判为狗，86 张狗被判为猫，两个方向合计占全部错例的 15.2%。错例图中标出了测试集编号，便于查找原图。猫狗类别激活图（CAM）将同一张图的两类响应并排显示，用统一的色标标出各区域对猫、狗分数的贡献。

[每类准确率](results/seed42/test/per_class.csv) · [混淆矩阵](results/seed42/test/confusion_matrix.png) · [六张错例](results/seed42/test/typical_errors.png) · [猫狗 CAM 对照](results/seed42/test/cat_dog_cam.png)

## 重新训练或绘图

`runs/` 保存四组的配置、划分索引和逐轮记录。以下命令沿用原始划分，将本次训练的权重和记录写入 `retrain/` 下的四个新目录。再次训练时，请为每组指定新的输出目录。

```powershell
$py = ".\.venv\Scripts\python.exe"
$split = "runs/baseline_seed42/split.npz"
& $py train.py --split $split --out retrain/baseline_seed42
& $py train.py --split $split --scheduler cosine --out retrain/cosine_seed42
& $py train.py --split $split --augment --out retrain/augment_seed42
& $py train.py --split $split --augment --scheduler cosine --out retrain/augment_cosine_seed42
& $py compare_runs.py --runs retrain --out verification/retrained
```

训练曲线和四组验证结果可直接从仓库保存的训练记录生成。完成测试集评估后，最后两条命令会读取 `verification/test/` 中的预测，绘制混淆矩阵、典型错例和猫狗类别激活图。

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py plot_learning.py
& $py compare_runs.py
& $py plot_test.py verification/test
& $py cat_dog_cam.py --result verification/test
```
