"""
独立评测脚本：加载已保存的 checkpoint，在测试集上计算 Loss / PPL / BLEU，
并导出详细的翻译对照结果。

用法:
    python evaluate.py --ckpt ckpt_dotproduct.pt --test eng-fra_test_data.txt --out test_result_dotproduct.txt
    python evaluate.py --ckpt ckpt_additive.pt   --test eng-fra_test_data.txt --out test_result_additive.txt

脚本会根据 ckpt 里的 attention_type 字段自动选择要实例化的 Transformer 结构：
    - "dotproduct" -> 从 transformer_dotproduct.py 导入
    - "additive"   -> 从 transformer_additive.py   导入
因此保持这三个脚本在同一目录即可。
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def build_vocab(itos):
    class V:
        def __init__(self, itos):
            self.itos = list(itos)
            self.stoi = {tok: i for i, tok in enumerate(itos)}

        def encode(self, s, add_sos_eos=False):
            ids = [self.stoi.get(tok, 3) for tok in s.split()]
            if add_sos_eos:
                ids = [1] + ids + [2]
            return ids

        def decode(self, ids):
            toks = []
            for i in ids:
                if i == 2:
                    break
                if i in (0, 1):
                    continue
                toks.append(self.itos[i])
            return " ".join(toks)

        def __len__(self):
            return len(self.itos)

    return V(itos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径 (ckpt_*.pt)")
    parser.add_argument("--test", required=True, help="测试集路径 (eng \\t fra)")
    parser.add_argument("--out", default="test_result.txt", help="详细结果输出文件")
    parser.add_argument("--max_len", type=int, default=20, help="训练时的句长过滤阈值")
    parser.add_argument("--decode_max_len", type=int, default=30)
    parser.add_argument("--max_eval", type=int, default=2000,
                        help="贪心解码 + BLEU 使用的前 N 条（整表 loss 仍用全集）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    attn_type = ckpt["attention_type"]
    print(f"Loaded checkpoint: attention_type={attn_type}")

    # 按 attention_type 动态导入
    if attn_type == "dotproduct":
        from transformer_dotproduct import (
            Transformer, load_pairs, TranslationDataset, collate_fn,
            evaluate, greedy_translate, corpus_bleu, PAD_IDX,
        )
    elif attn_type == "additive":
        from transformer_additive import (
            Transformer, load_pairs, TranslationDataset, collate_fn,
            evaluate, greedy_translate, corpus_bleu, PAD_IDX,
        )
    else:
        raise ValueError(f"Unknown attention_type: {attn_type}")

    # 复原词表
    src_vocab = build_vocab(ckpt["src_itos"])
    tgt_vocab = build_vocab(ckpt["tgt_itos"])

    # 构建模型（从 ckpt['args'] 取超参）
    sa = ckpt["args"]
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=sa.get("d_model", 256),
        num_heads=sa.get("num_heads", 8),
        num_layers=sa.get("num_layers", 4),
        d_ff=sa.get("d_ff", 1024),
        dropout=sa.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 加载测试集
    pairs = load_pairs(args.test, max_len=args.max_len)
    print(f"Test pairs (after filtering len<= {args.max_len}): {len(pairs)}")

    # token-level loss/PPL
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    ds = TranslationDataset(pairs, src_vocab, tgt_vocab)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loss = evaluate(model, loader, criterion, device)
    test_ppl = math.exp(min(test_loss, 20))

    # BLEU
    eval_pairs = pairs[: args.max_eval]
    hyps, refs = [], []
    for en, fr in eval_pairs:
        hyps.append(greedy_translate(model, en, src_vocab, tgt_vocab, device, max_len=args.decode_max_len))
        refs.append(fr)
    bleu = corpus_bleu(hyps, refs)

    # 打印汇总
    header = (
        f"=== Evaluation Summary ===\n"
        f"ckpt              : {args.ckpt}\n"
        f"attention_type    : {attn_type}\n"
        f"test_file         : {args.test}\n"
        f"num test pairs    : {len(pairs)}\n"
        f"BLEU sample size  : {len(eval_pairs)}\n"
        f"Test Loss         : {test_loss:.4f}\n"
        f"Test PPL          : {test_ppl:.2f}\n"
        f"Corpus BLEU (1-4) : {bleu:.2f}\n"
    )
    print(header)

    # 写入详细翻译对照
    lines = [header, "=== Translations (first 100) ==="]
    for (en, ref), hyp in zip(eval_pairs[:100], hyps[:100]):
        lines.append(f"EN : {en}\nREF: {ref}\nHYP: {hyp}\n")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Detailed results written to: {args.out}")


if __name__ == "__main__":
    main()
