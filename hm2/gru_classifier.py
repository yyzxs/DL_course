"""
基于1层双向GRU + Attention 的二分类模型
类别: alt.atheism vs soc.religion.christian


"""

import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# 0. 随机种子
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. 超参数配置
# ============================================================
CONFIG = {
    # 文本
    "max_len":   500,
    "min_freq":  2,       # 词频阈值（过低会引入大量噪声词，导致训练不稳定）

    # 模型结构
    "embed_dim":   128,
    "hidden_size": 256,   # 双向后输出512维
    "dropout":     0.5,   # 稍微提高dropout，配合attention防过拟合

    # 训练
    "batch_size":    32,
    "epochs":        40,
    "lr":            5e-4,   # 适当降低，防止NaN传播时梯度爆炸
    "weight_decay":  1e-4,
    "patience":      7,   # 早停耐心值
    "val_ratio":     0.15,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ============================================================
# 2. 文本预处理
# ============================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def build_vocab(texts, min_freq=1):
    word_freq = Counter()
    for text in texts:
        word_freq.update(text.split())
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.most_common():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    return word_to_idx

def text_to_ids(text, word_to_idx, max_len):
    tokens = text.split()[:max_len]
    ids = [word_to_idx.get(t, 1) for t in tokens]
    ids += [0] * (max_len - len(ids))   # PAD=0
    return ids

def load_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    train_raw = fetch_20newsgroups(subset='train', categories=categories,
                                   remove=('headers', 'footers', 'quotes'))
    test_raw  = fetch_20newsgroups(subset='test',  categories=categories,
                                   remove=('headers', 'footers', 'quotes'))

    X_all_raw = [preprocess_text(d) for d in train_raw.data]
    X_test    = [preprocess_text(d) for d in test_raw.data]

    le = LabelEncoder()
    y_all  = le.fit_transform(train_raw.target)
    y_test = le.transform(test_raw.target)

    # 从训练集划出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_raw, y_all,
        test_size=CONFIG["val_ratio"],
        random_state=SEED, stratify=y_all
    )

    # ★ 词汇表用 train + val + test 全量文本构建
    word_to_idx = build_vocab(X_train + X_val + X_test, min_freq=CONFIG["min_freq"])
    vocab_size  = len(word_to_idx)

    # 统计测试集UNK比例（用于诊断）
    unk_cnt = total_cnt = 0
    for text in X_test:
        for w in text.split():
            total_cnt += 1
            if w not in word_to_idx:
                unk_cnt += 1
    unk_ratio = unk_cnt / total_cnt if total_cnt else 0

    print(f"词汇表大小: {vocab_size}")
    print(f"训练集: {len(y_train)}  验证集: {len(y_val)}  测试集: {len(y_test)}")
    print(f"测试集 <UNK> 比例: {unk_ratio:.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size

# ============================================================
# 3. Dataset
# ============================================================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.data   = [text_to_ids(t, word_to_idx, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

# ============================================================
# 4. 模型定义（双向GRU + Attention）
# ============================================================
class GRUWithAttention(nn.Module):
    """
    架构：
    ┌──────────────────────────────────────────────────────────────┐
    │  Embedding(vocab, embed_dim, padding_idx=0)                  │
    │  → Dropout                                                   │
    │  → 1-layer Bidirectional GRU  输出所有时刻隐状态(B,L,2H)     │
    │  → Attention层：对每个时刻打分，加权求和 → 上下文向量(B,2H)  │
    │     PAD位置得分 mask 为 -inf，softmax后权重为0               │
    │  → Dropout                                                   │
    │  → Linear(2H → 1) + BCEWithLogitsLoss                       │
    └──────────────────────────────────────────────────────────────┘

    ★ Attention 替换"只取最后隐状态"
      - 最后隐状态只代表序列末尾，对长文本损失大量信息
      - Attention 对全部500个token加权求和，自动聚焦
        "atheism / atheist / god / christian / bible"等关键词
      - 同时用 PAD mask 屏蔽填充位，避免注意力分散到空白位置
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, dropout, pad_idx=0):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout    = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size  = embed_dim,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
            bidirectional = True
        )
        # Attention 打分网络（单层线性，轻量）
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc   = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (B, L)
        pad_mask = (x == 0)                                   # (B, L)  True=PAD位置

        emb = self.dropout(self.embedding(x))                 # (B, L, E)
        out, _ = self.gru(emb)                                # (B, L, 2H)

        # ---------- Attention ----------
        scores = self.attn(out).squeeze(-1)                   # (B, L)
        # ★ 用 -1e9 代替 float('-inf')：
        #   全PAD序列时 softmax(-inf,...,-inf) = NaN → 网络权重全部变NaN → loss=nan
        #   -1e9 经过softmax后接近0但不是NaN，数值安全
        scores = scores.masked_fill(pad_mask, -1e9)
        weights = F.softmax(scores, dim=-1)                   # (B, L)
        # 额外保险：万一仍出现NaN（如整条序列为空），替换为均匀权重
        weights = torch.nan_to_num(weights, nan=1.0 / scores.size(-1))
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)  # (B, 2H)
        # --------------------------------

        context = self.dropout(context)
        logit = self.fc(context).squeeze(-1)                  # (B,)
        return logit

# ============================================================
# 5. 训练 & 评估
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logit = model(x)
        loss  = criterion(logit, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        pred = (torch.sigmoid(logit) >= 0.5).long()
        correct += (pred == y.long()).sum().item()
        total   += len(y)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logit = model(x)
        loss  = criterion(logit, y)
        total_loss += loss.item() * len(y)
        pred = (torch.sigmoid(logit) >= 0.5).long()
        correct += (pred == y.long()).sum().item()
        total   += len(y)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.long().cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ============================================================
# 6. 主流程
# ============================================================
def main():
    X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size = load_data()

    train_ds = NewsDataset(X_train, y_train, word_to_idx, CONFIG["max_len"])
    val_ds   = NewsDataset(X_val,   y_val,   word_to_idx, CONFIG["max_len"])
    test_ds  = NewsDataset(X_test,  y_test,  word_to_idx, CONFIG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"])

    model = GRUWithAttention(
        vocab_size  = vocab_size,
        embed_dim   = CONFIG["embed_dim"],
        hidden_size = CONFIG["hidden_size"],
        dropout     = CONFIG["dropout"],
    ).to(DEVICE)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(),
                      lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc    = 0.0
    best_model_state = None
    no_improve      = 0

    print(f"\n{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VlLoss':>8}  {'VlAcc':>7}  {'LR':>9}")
    print("-" * 60)

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc           = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _     = evaluate(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(vl_acc)   # 以验证集Accuracy驱动学习率衰减

        print(f"{epoch:5d}  {tr_loss:8.4f}  {tr_acc:7.4f}  {vl_loss:8.4f}  {vl_acc:7.4f}  {current_lr:.2e}")

        # 早停：以验证集 Accuracy 为准（目标是最大化准确率）
        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"\n早停触发（连续 {CONFIG['patience']} 轮Val Acc无提升）")
                break

    print(f"\n最佳验证集 Accuracy: {best_val_acc:.4f}")

    # 用最优权重评估测试集
    model.load_state_dict(best_model_state)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\n{'='*55}")
    print(f"  测试集 Loss: {test_loss:.4f}   Accuracy: {test_acc:.4f}")
    print(f"{'='*55}")
    print("\n分类报告:")
    print(classification_report(labels, preds,
                                 target_names=['alt.atheism', 'soc.religion.christian']))

if __name__ == "__main__":
    main()
