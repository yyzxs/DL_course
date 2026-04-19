"""
基于 BERT 的二分类模型
类别: alt.atheism vs soc.religion.christian
对比对象: 作业2中的 1层双向GRU + Attention 模型
"""

import re
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from transformers import BertTokenizer, BertModel
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
    "model_name":   "bert-base-uncased",
    "max_len":      128,      # BERT 上限512，128已覆盖大部分文本且速度快
    "batch_size":   16,       # BERT 显存占用大，batch不宜过大
    "epochs":       5,        # 微调不需要太多轮
    "lr":           2e-5,     # BERT 微调标准学习率（1e-5 ~ 5e-5）
    "weight_decay": 1e-2,
    "warmup_ratio": 0.1,      # 前10%步数线性warmup，防止初期梯度过大
    "patience":     3,        # 早停耐心值
    "val_ratio":    0.15,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ============================================================
# 2. 数据加载
# ============================================================
def preprocess_text(text):
    """与 20_news_data.py 保持一致"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def load_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    train_raw = fetch_20newsgroups(subset='train', categories=categories,
                                   remove=('headers', 'footers', 'quotes'))
    test_raw  = fetch_20newsgroups(subset='test',  categories=categories,
                                   remove=('headers', 'footers', 'quotes'))

    X_all = [preprocess_text(d) for d in train_raw.data]
    X_test = [preprocess_text(d) for d in test_raw.data]

    le = LabelEncoder()
    y_all  = le.fit_transform(train_raw.target)
    y_test = le.transform(test_raw.target)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=CONFIG["val_ratio"],
        random_state=SEED, stratify=y_all
    )

    print(f"训练集: {len(y_train)}  验证集: {len(y_val)}  测试集: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================
# 3. Dataset（使用 BERT Tokenizer）
# ============================================================
class BertNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'label':          self.labels[idx]
        }

# ============================================================
# 4. 模型定义
# ============================================================
class BertClassifier(nn.Module):
    """
    架构：
    ┌────────────────────────────────────────────────────────┐
    │  BERT (bert-base-uncased, 12层, 768维, 1.1亿参数)      │
    │  取 [CLS] token 的最终隐状态作为句子表示               │
    │  → Dropout(0.1)                                        │
    │  → Linear(768 → 1)                                     │
    │  → BCEWithLogitsLoss                                   │
    └────────────────────────────────────────────────────────┘

    与 GRU 的核心区别：
    - BERT 是预训练模型，已在 BookCorpus + Wikipedia 上学到丰富语言知识
    - 双向 Transformer Self-Attention 对全序列建模，无遗忘问题
    - 微调（Fine-tuning）只需少量数据和轮次即可获得高精度
    """
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert    = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # pooler_output: [CLS] token 经过线性层+tanh后的输出，专为分类任务设计
        cls = outputs.pooler_output           # (B, 768)
        cls = self.dropout(cls)
        logit = self.fc(cls).squeeze(-1)      # (B,)
        return logit

# ============================================================
# 5. 训练 & 评估
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        y              = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logit = model(input_ids, attention_mask, token_type_ids)
        loss  = criterion(logit, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

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
    for batch in loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        y              = batch['label'].to(DEVICE)

        logit = model(input_ids, attention_mask, token_type_ids)
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
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print(f"\n加载 Tokenizer: {CONFIG['model_name']}")
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = BertNewsDataset(X_train, y_train, tokenizer, CONFIG["max_len"])
    val_ds   = BertNewsDataset(X_val,   y_val,   tokenizer, CONFIG["max_len"])
    test_ds  = BertNewsDataset(X_test,  y_test,  tokenizer, CONFIG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"])

    model = BertClassifier(CONFIG["model_name"]).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    optimizer = AdamW(model.parameters(),
                      lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()

    # Warmup + 常数 LR 调度
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps]
    )

    best_val_acc     = 0.0
    best_model_state = None
    no_improve       = 0

    print(f"\n{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VlLoss':>8}  {'VlAcc':>7}")
    print("-" * 48)

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc       = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)

        print(f"{epoch:5d}  {tr_loss:8.4f}  {tr_acc:7.4f}  {vl_loss:8.4f}  {vl_acc:7.4f}")

        if vl_acc > best_val_acc:
            best_val_acc     = vl_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"\n早停触发（连续 {CONFIG['patience']} 轮 Val Acc 无提升）")
                break

    print(f"\n最佳验证集 Accuracy: {best_val_acc:.4f}")

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
