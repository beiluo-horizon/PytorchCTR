# PytorchCTR
Reproduce the classic and open source ctr model

复现了一些经典开源ctr模型，并持续完善

The code heavily references the Deepctr project for learning purposes only

代码大量参考了DeepCTR项目，仅做学习所用

DeepCTR URL：https://github.com/shenweichen/DeepCTR-Torch

The reproduction performance did not meet the results claimed in the paper

复现性能没有达到论文中宣称的结果

Possible reasons for code bugs or inconsistent data

可能是代码存在Bug或者数据不一致的原因


已复现模型（Reproduced model）：

|模型名（model name）|论文地址（paper address）|
|----|----|
|Wide&Deep|https://arxiv.org/pdf/1606.07792v1.pdf|
|DeepFM|https://arxiv.org/pdf/1703.04247v1.pdf|
|xDeepFM|https://arxiv.org/pdf/1803.05170v3.pdf|
|DCN|https://arxiv.org/pdf/1708.05123.pdf|
|AutoInt|https://arxiv.org/pdf/1810.11921v2.pdf|
|AFN|https://arxiv.org/pdf/1909.03276v2.pdf|
|GateNet|https://arxiv.org/pdf/2007.03519v1.pdf|
|FiBiNet|https://arxiv.org/pdf/1905.09433v1.pdf|
|FATDeepFFM|https://arxiv.org/pdf/1905.06336v1.pdf|
|FiBiNetPlus|https://arxiv.org/pdf/2209.05016v1.pdf|
|ContextNet|https://arxiv.org/pdf/2107.12025v1.pdf|
|DCNv2|https://arxiv.org/pdf/2008.13535v2.pdf|
|MaskNet|https://arxiv.org/pdf/2102.07619v2.pdf|
|FinalMLP|https://arxiv.org/pdf/2304.00902v3.pdf|




运行流程   running process

第一步     step1

数据准备   prepare training data

```python
$ cd data/criteo
$ python download_criteo_x1.py
$ python trans.py (please modify the path accordingly)
```

第二步     step2

提前在config中配置好需要的模型和参数   Configure the required models and parameters in advance in config


第三步     step3

在main.py中配置模型设置和参数设置   configure model and parameter settings in main.py

以FinalMLP为例      Taking FinalMLP as an example

```python
$ python mian.py --config_dir='./config/FinalMLP_criteo_x1' --model_setid='base' data_setid='base' gpu_index=0 expid='v1'
```
