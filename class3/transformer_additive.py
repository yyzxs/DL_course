"""
基于 Transformer 的英法机器翻译 —— 加性（Bahdanau）注意力版本
Additive (Bahdanau) Attention Transformer for English -> French Translation

与 transformer_dotproduct.py 的唯一核心区别在 AdditiveAttention 类：
    Attention(Q, K, V) = softmax( v^T * tanh(W_q Q + W_k K) ) V

其他结构（多头、残差、位置编码、训练流程）保持一致，以保证公平对比。

Usage:
    python transformer_additive.py --data eng-fra_train_data.txt --epochs 10
"""

import argparse
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# -----------------------------
# 1. 特殊标记 & 数据预处理（与点积版完全一致）
# -----------------------------
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def normalize_string(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"([.!?,¿¡])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z0-9àâäéèêëïîôöùûüÿçœæ.!?,¿¡'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    return s.split()


def load_pairs(path: str, max_len: int = 20, max_pairs: int = None):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eng, fra = parts[0], parts[1]
            eng, fra = normalize_string(eng), normalize_string(fra)
            if not eng or not fra:
                continue
            if len(tokenize(eng)) > max_len or len(tokenize(fra)) > max_len:
                continue
            pairs.append((eng, fra))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


class Vocab:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.itos = list(SPECIAL_TOKENS)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, sentences):
        counter = Counter()
        for s in sentences:
            counter.update(tokenize(s))
        for word, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def encode(self, s: str, add_sos_eos: bool = False):
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokenize(s)]
        if add_sos_eos:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids):
        toks = []
        for i in ids:
            if i == EOS_IDX:
                break
            if i in (PAD_IDX, SOS_IDX):
                continue
            toks.append(self.itos[i])
        return " ".join(toks)

    def __len__(self):
        return len(self.itos)


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = torch.tensor(self.src_vocab.encode(src, add_sos_eos=True), dtype=torch.long)
        tgt_ids = torch.tensor(self.tgt_vocab.encode(tgt, add_sos_eos=True), dtype=torch.long)
        return src_ids, tgt_ids


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
    tgts = pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX)
    return srcs, tgts


# -----------------------------
# 2. 位置编码
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# -----------------------------
# 3. 加性 (Bahdanau) 注意力 —— 核心改动
# -----------------------------
class AdditiveAttention(nn.Module):
    """
    Additive / Bahdanau Attention:
        e_{ij} = v^T * tanh( W_q * q_i + W_k * k_j )
        alpha = softmax(e)
        out = alpha * V

    为了支持多头形式，每个 head 内都使用一组独立的 (W_q, W_k, v)。
    采用"广播加法 + 投影为标量"的高效向量化实现。
    """

    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        # 对 Q / K 再做一次对齐线性映射，得到打分用的隐藏向量
        self.W_q = nn.Linear(d_k, d_k, bias=False)
        self.W_k = nn.Linear(d_k, d_k, bias=False)
        # 打分向量 v: [d_k] -> 标量
        self.v = nn.Linear(d_k, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q: [B, h, L_q, d_k]   K,V: [B, h, L_k, d_k]
        # 映射: [B, h, L_q, d_k] / [B, h, L_k, d_k]
        q_proj = self.W_q(Q).unsqueeze(-2)          # [B, h, L_q, 1,   d_k]
        k_proj = self.W_k(K).unsqueeze(-3)          # [B, h, 1,   L_k, d_k]
        # 广播相加 -> [B, h, L_q, L_k, d_k]
        energy = torch.tanh(q_proj + k_proj)
        # 投影为标量得分 -> [B, h, L_q, L_k]
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out, attn


class MultiHeadAdditiveAttention(nn.Module):
    """多头加性注意力（替换原多头缩放点积注意力）"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 每个头共享的加性打分模块（内部已经按 d_k 展开）
        self.attention = AdditiveAttention(self.d_k, dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B,1,L_q,L_k]

        out, _ = self.attention(Q, K, V, mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# -----------------------------
# 4. 前馈网络 & 编码/解码层
# -----------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAdditiveAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAdditiveAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAdditiveAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# -----------------------------
# 5. Transformer 整体结构
# -----------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_len=100,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def make_src_mask(src):
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt):
        B, L = tgt.size()
        pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(torch.ones((L, L), device=tgt.device)).bool()
        return pad_mask & sub_mask

    def encode(self, src, src_mask):
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.out_proj(dec_out)


# -----------------------------
# 6. 训练 & 推理
# -----------------------------
def train_epoch(model, loader, optim, criterion, device, clip=1.0):
    model.train()
    total_loss, n_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()

        non_pad = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * non_pad
        n_tokens += non_pad
    return total_loss / max(n_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        non_pad = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * non_pad
        n_tokens += non_pad
    return total_loss / max(n_tokens, 1)


@torch.no_grad()
def greedy_translate(model, sentence, src_vocab, tgt_vocab, device, max_len=30):
    model.eval()
    src_ids = torch.tensor(
        [src_vocab.encode(normalize_string(sentence), add_sos_eos=True)],
        dtype=torch.long,
        device=device,
    )
    src_mask = model.make_src_mask(src_ids)
    enc_out = model.encode(src_ids, src_mask)

    ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(ys)
        out = model.decode(ys, enc_out, src_mask, tgt_mask)
        prob = model.out_proj(out[:, -1])
        next_id = prob.argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == EOS_IDX:
            break
    return tgt_vocab.decode(ys[0].tolist()[1:])


# -----------------------------
# 6.x  简易 corpus-level BLEU (1-4 gram + brevity penalty)
# -----------------------------
def _ngram_counts(tokens, n):
    from collections import Counter
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses, references, max_n: int = 4):
    clipped = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        h, r = hyp.split(), ref.split()
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hc, rc = _ngram_counts(h, n), _ngram_counts(r, n)
            for ng, c in hc.items():
                clipped[n - 1] += min(c, rc.get(ng, 0))
            total[n - 1] += max(len(h) - n + 1, 0)
    precisions = [(clipped[n] + 1e-9) / (total[n] + 1e-9) for n in range(max_n)]
    if min(precisions) <= 0:
        return 0.0
    log_p = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(log_p) * 100.0


@torch.no_grad()
def evaluate_testset(model, test_pairs, src_vocab, tgt_vocab, device, criterion=None, max_len=30, max_eval=None):
    model.eval()
    ds = TranslationDataset(test_pairs, src_vocab, tgt_vocab)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    loss = float("nan")
    if criterion is not None:
        loss = evaluate(model, loader, criterion, device)

    eval_pairs = test_pairs if max_eval is None else test_pairs[:max_eval]
    hyps, refs = [], []
    for src, tgt in eval_pairs:
        pred = greedy_translate(model, src, src_vocab, tgt_vocab, device, max_len=max_len)
        hyps.append(pred)
        refs.append(tgt)
    bleu = corpus_bleu(hyps, refs)
    return loss, bleu, list(zip([p[0] for p in eval_pairs], refs, hyps))


# -----------------------------
# 7. main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="eng-fra_train_data.txt")
    parser.add_argument("--max_pairs", type=int, default=30000)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="train_log_additive.txt")
    parser.add_argument("--test", type=str, default=None, help="test set path, format: eng\\tfra")
    parser.add_argument("--test_max_eval", type=int, default=2000,
                        help="只用前 N 条做贪心解码与 BLEU (整表 loss 仍用全集)")
    parser.add_argument("--ckpt", type=str, default="ckpt_additive.pt",
                        help="保存模型与词表的路径 (便于 evaluate.py 后续加载)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = load_pairs(args.data, max_len=args.max_len, max_pairs=args.max_pairs)
    random.shuffle(pairs)
    n_val = max(1, len(pairs) // 20)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    src_vocab, tgt_vocab = Vocab(min_freq=2), Vocab(min_freq=2)
    src_vocab.build([p[0] for p in train_pairs])
    tgt_vocab.build([p[1] for p in train_pairs])
    print(f"Src vocab: {len(src_vocab)}  Tgt vocab: {len(tgt_vocab)}")

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    log_lines = ["Epoch\tTrainLoss\tValLoss\tValPPL\tTime(s)"]
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optim, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        ppl = math.exp(min(val_loss, 20))
        line = f"{epoch}\t{tr_loss:.4f}\t{val_loss:.4f}\t{ppl:.2f}\t{dt:.1f}"
        print(f"[Additive]  Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f} | ppl {ppl:.2f} | {dt:.1f}s")
        log_lines.append(line)

    Path(args.log_file).write_text("\n".join(log_lines), encoding="utf-8")

    # 保存 checkpoint
    torch.save(
        {
            "model_state": model.state_dict(),
            "src_itos": src_vocab.itos,
            "tgt_itos": tgt_vocab.itos,
            "args": vars(args),
            "attention_type": "additive",
        },
        args.ckpt,
    )
    print(f"Checkpoint saved to: {args.ckpt}")

    if args.test:
        print(f"\n=== Evaluating on test set: {args.test} ===")
        test_pairs = load_pairs(args.test, max_len=args.max_len)
        print(f"Test pairs (after filtering): {len(test_pairs)}")
        test_loss, bleu, samples = evaluate_testset(
            model, test_pairs, src_vocab, tgt_vocab, device,
            criterion=criterion, max_eval=args.test_max_eval,
        )
        test_ppl = math.exp(min(test_loss, 20))
        print(f"[Additive]  Test Loss={test_loss:.4f}  PPL={test_ppl:.2f}  BLEU={bleu:.2f}")
        lines = [f"# Test Loss={test_loss:.4f}  PPL={test_ppl:.2f}  BLEU={bleu:.2f}"]
        for en, ref, hyp in samples[:30]:
            lines.append(f"EN:  {en}\nREF: {ref}\nHYP: {hyp}\n")
        Path(args.log_file.replace(".txt", "_test.txt")).write_text("\n".join(lines), encoding="utf-8")

    fixed = ["i am happy .", "he is a good boy .", "this is my book .", "she loves cats ."]
    print("\n=== Sample Translations (Additive) ===")
    for s in fixed:
        print(f"  EN: {s}")
        print(f"  FR: {greedy_translate(model, s, src_vocab, tgt_vocab, device)}\n")


if __name__ == "__main__":
    main()
