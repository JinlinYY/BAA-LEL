# BUA-LEL 使用说明

本仓库面向科研人员，提供边界不确定性感知的病灶证据学习方法 BUA-LEL，包括模型、分割与分类基线、训练和推理、消融实验、区域参数敏感性、校准、配对统计比较及可解释性分析。

## 使用步骤

1. 安装与设备匹配的 PyTorch/torchvision，再执行 `pip install -r requirements.txt` 和 `pip install -e . --no-deps`。
2. 按 [数据说明](data.md) 在本地准备图像、掩膜和临床表格，自行获取 MedSAM 权重。
3. 修改 `configs/` 中的数据与权重路径。
4. 先运行 `python scripts/cross_validate.py --config configs/her2usc.yaml --dry-run` 检查配置，再去掉 `--dry-run` 开始训练。
5. 使用 `scripts/inference.py` 进行无需标签的单病例推理，使用 `scripts/evaluate.py` 在指定留出病例上评估。

主要模块术语与论文对应：病灶先验感知语义特征编码、锚点约束边界图、不确定性校准边缘几何编码、尺度自适应分区形态编码、形态—临床异构图推理。模块路径与实现细节见 [模型说明](model.md)。

## 数据与复现范围

HER2USC、LMNUSC 为私有临床队列；BrEaST 用于联合分割和二分类；BUSI 用于纯图像分割。根据作者确认，所提供论文中标为 ISIC2018 的跨域实验实际使用 IMA++。因此，本仓库以 `imaplusplus.yaml` 和 `imaplusplus_multiclass.yaml` 表示对应数据，`isic2018.yaml` 单独用于 ISIC2018 分割，不能互换。

本仓库不包含数据集、临床记录、病例划分、训练结果、论文 PDF 或模型权重。公开配置是可运行的研究入口，不代表已经重新训练并验证了论文所有表格数值。主训练器的折内验证选模与基线的内外层划分属于不同评估协议，不能直接混合解释。IMA++ 的现有实验配置使用 ResNet-50，与论文通用设置中的 MedSAM 不同。详见 [实验协议](experiments.md)。

临床图沿用现有科学实现：完整临床向量用于初始化 12 个先验节点，并非每个节点只输入一项临床变量。该细节对解释图节点作用和迁移到其他队列非常重要。

自有代码采用 MIT 许可证；第三方 Segment Anything 保留 Apache-2.0。请按 `CITATION.cff` 引用软件和对应论文。
