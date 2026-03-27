# 基于 GRU + Attention 的新闻文本二分类

使用 PyTorch 实现的文本分类项目，在 20 Newsgroups 数据集的两个类别（无神论 vs 基督教）上完成二分类任务。

---

## 项目结构

```
.
├── 20_news_data.py       # 数据加载与预处理模块
├── gru_classifier.py     # 模型定义、训练与评估主程序
└── README.md
```

---

## 任务说明

| 项目 | 内容 |
|------|------|
| 数据集 | [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) |
| 分类类别 | `alt.atheism`（无神论）vs `soc.religion.christian`（基督教） |
| 任务类型 | 二分类（Binary Classification） |
| 移除字段 | headers、footers、quotes（避免信息泄露） |

---

## 模型架构

```
输入序列 (B, L)
    │
    ▼
Embedding Layer  [vocab_size × 128]
    │
    ▼  Dropout(0.5)
    │
    ▼
1-layer Bidirectional GRU  [hidden=256, 输出维度=512]
    │
    ▼  对所有时刻输出打分
Attention Layer  [512 → 1]  +  PAD Mask
    │  softmax → 加权求和
    ▼
Context Vector  [512]
    │
    ▼  Dropout(0.5)
    │
    ▼
Linear  [512 → 1]
    │
BCEWithLogitsLoss
```

**双向 GRU** 同时从左到右、从右到左读取序列，拼接两个方向的输出，捕获完整上下文。

**Attention 机制** 对序列中每个位置的 GRU 输出计算重要性分数，自动聚焦"atheism / christian / scripture"等判别性词汇，而不是只依赖末尾隐状态。PAD 位置的分数被置为 `-1e9`，避免注意力权重出现 NaN。

---

## 环境依赖

```bash
pip install torch scikit-learn numpy
```

| 库 | 推荐版本 |
|----|---------|
| Python | >= 3.8 |
| PyTorch | >= 2.0 |
| scikit-learn | >= 1.0 |
| numpy | >= 1.21 |

---

## 快速开始

```bash
python gru_classifier.py
```

训练过程会打印每轮的 Loss 和 Accuracy，验证集准确率连续 7 轮无提升时自动早停，最终输出测试集评估报告。

---

## 超参数配置

所有超参数集中在 `gru_classifier.py` 顶部的 `CONFIG` 字典中，修改后直接重新运行即可。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_len` | 500 | 序列截断/填充长度 |
| `min_freq` | 2 | 词汇表最低词频阈值 |
| `embed_dim` | 128 | 词向量维度 |
| `hidden_size` | 256 | GRU 隐藏层大小（双向后输出 512） |
| `dropout` | 0.5 | Dropout 比例 |
| `batch_size` | 32 | 批大小 |
| `epochs` | 40 | 最大训练轮数 |
| `lr` | 5e-4 | AdamW 学习率 |
| `weight_decay` | 1e-4 | L2 正则化系数 |
| `patience` | 7 | 早停耐心值 |
| `val_ratio` | 0.15 | 验证集划分比例 |

---

## 训练策略

- **优化器**：AdamW，解耦权重衰减，适合 NLP 任务
- **学习率调度**：`ReduceLROnPlateau`，验证集准确率连续 3 轮无提升则学习率 × 0.5
- **早停**：监控验证集准确率，连续 7 轮无提升停止训练
- **梯度裁剪**：`max_norm=1.0`，防止梯度爆炸
- **词汇表**：由训练集 + 验证集 + 测试集全量文本构建，避免测试集词汇大量变为 `<UNK>`

---

## 注意事项

1. 首次运行会自动从网络下载 20 Newsgroups 数据集（约 14 MB），需要网络连接。
2. 若 GPU 可用，程序会自动切换到 CUDA 运行，CPU 上训练约需 2~5 分钟。
3. Attention 的 PAD mask 使用 `-1e9` 而非 `-inf`，防止全空文档导致 `softmax` 输出 NaN。
