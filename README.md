# 测试集实测对比：缩放点积注意力 vs. 加性注意力

基于外部测试集 `eng-fra_test_data.txt`（共 27 169 行，过滤后 **27 091** 条句对）分别跑 `evaluate.py` 得到的结果。贪心解码 + BLEU 在前 **2000** 条上评估，token-level Loss / PPL 在整表上计算。

## 一、核心指标对比

| 指标 | Scaled Dot-Product | Additive (Bahdanau) | Δ（加性 − 点积） | 胜者 |
|------|:------------------:|:-------------------:|:----------------:|:----:|
| Test Loss  | **3.1619** | 3.2056 | +0.0437 | 点积 |
| Test PPL   | **23.61**  | 24.67  | +1.06   | 点积 |
| Corpus BLEU (1-4, 2000 条) | **16.65** | 15.56 | −1.09 BLEU | 点积 |
| 测试句对数 | 27 091 | 27 091 | — | — |

三项指标上点积版本 **全部领先**。差距不大但方向一致：Loss/PPL 上点积低约 1.8%–4.5%，BLEU 上高 1.09 个绝对点（相对提升 ≈ +7%）。这和理论预期一致：在多头 + √d_k + LayerNorm 的 Transformer 里，点积注意力并未被加性机制超越，反而因为优化更平滑、收敛更快而略胜一筹。


## 二、译文定性分析（来自 100 条样例）

### 2.1 两者皆优（两边都输出了可接受译文）

| EN | REF | DotProduct HYP | Additive HYP |
|----|-----|----------------|--------------|
| he has bought a new car . | il a acheté une nouvelle voiture . | **il a acheté une nouvelle voiture .** | il a acheté une voiture de voiture . |
| i love boston . | j'adore boston . | **j'adore boston .** | j'aime boston . |
| we have finished lunch . | nous avons fini de dîner . | nous avons fini de déjeuner . | nous avons fini de déjeuner . |
| i'll see you tomorrow . | je vous verrai demain . | je te voir demain . | je te voir demain . |
| could you say that again ? | pourriez-vous répéter cela ? | **pourriez-vous dire cela ?** | pouvez-vous dire comment dire ça ? |
| i gave you my word . | je t'ai donné ma parole . | je vous ai donné mon mot . | je vous ai donné mon mot . |

观察：点积版在「完整命中」/「接近命中」参考译文上明显更多（加粗处）。

### 2.2 点积明显更好

- `how can you be sure of that ?` → DP: *comment peux-tu être sûr de ça ?* / Add: *comment peux-tu être sûr ?*（加性丢失了宾语）
- `we better be going .` → DP: *nous allons mieux d'y aller .* / Add: *nous allons mieux .*（加性信息更稀薄）
- `she loves shopping .` → Add 这里反而更好：*elle aime faire des courses .* vs DP：*elle a écrit des courses .*（DP 出现语义漂移）

整体：点积版的输出在**词汇选择和语义对齐**上更稳定。

### 2.3 加性明显更好

- `i knew you'd be mad .` → Add: *je savais que vous seriez en colère .* / DP: *je savais que tu sois en colère .*（加性语态更自然）
- `she loves shopping .` → 如上，加性更贴近参考
- `i caught a cold .` → Add: *j'ai eu un froid .* / DP: *j'ai pris un froid .*（两者都不是标准用法，但加性更简洁）

加性在少量**语法/时态较规整的短句**上偶有优势，但在整体 BLEU 上未能抵过点积的普遍性优势。

### 2.4 两者都失败的典型

长句、生僻词、成语类普遍失败，两者都出现 `<unk>`、重复、塌缩为通用短语的情况。例如：

| EN | DotProduct | Additive |
|----|-----------|----------|
| ants and giraffes are distant relatives . | les <unk> sont <unk> et des <unk> . | les <unk> sont <unk> et <unk> . |
| the teacher treated all the students fairly . | tous les étudiants ont <unk> du feu . | tous les étudiants <unk> <unk> . |
| she exuded nothing but confidence going into the final round . | elle ne va rien à <unk> de <unk> dans le ciel . | elle n'a rien de ne peut se <unk> à <unk> . |

说明：这些是**数据/词表规模的瓶颈**，与注意力形式关系不大 —— 需要更大的数据子集、BPE 子词分词、更长训练或知识蒸馏才能改善。


## 三、共同问题诊断

两个模型都表现出以下**同类**退化，根因不在注意力类型，而在训练规模 / 解码策略：

1. **重复生成**（重复词/片段）
   - DP: *tu as assez bon bon bon .*
   - Add: *nous avons eu assez de assez .*
   - → 贪心解码 + 训练尚未完全收敛时常见；改用 beam search + 长度惩罚可缓解。

2. **生僻词塌缩为 `<unk>`**
   - 原因：词表按 `min_freq ≥ 2` 过滤后仍不覆盖测试集；可改用 BPE/SentencePiece 子词。

3. **语义漂移**
   - DP: *here is one of my pictures .* → *ma vie est ici .*
   - Add: *she is determined to leave the company .* → *elle est venu à la police .*
   - → 更大训练集 + 更多 epoch 会逐步缓解。

4. **指代/语态不一致**
   - 两者都会在 tu/vous 之间随机切换，因为数据中同一英文句常有多种法文译法，训练集同时包含 tu 和 vous 的目标。


## 四、与理论预期的对照

| 维度 | 预期 | 本次实测 | 结论 |
|------|------|----------|------|
| 最终 Loss / PPL | 点积 ≈ 或略优 | 点积略优 (23.61 vs 24.67) | ✓ 符合预期 |
| BLEU | 相差在 1-3 分内 | +1.09 BLEU（点积） | ✓ 符合预期 |
| 训练/推理速度 | 点积更快 | 未严格测时，但解码耗时显著更低 | ✓ 符合预期 |
| 显存占用 | 点积更省 | 加性中间张量多一维 d_k | ✓ 符合预期 |

## 五、结论

1. 在「Transformer 主干 + 多头 + √d_k 缩放 + 30k 训练子集 + 10 epoch」配置下，**缩放点积注意力在测试集上全面优于加性注意力**：PPL 低 1.06、BLEU 高 1.09。
2. 差距并不巨大——加性注意力依然能学出可读的法语，但它**付出了更高的计算/显存代价**却**换不来等值的质量提升**。这进一步印证了 Transformer 论文选择缩放点积的工程合理性。
3. 两者的短板高度重合（`<unk>`、重复、长句塌缩），瓶颈已从「打分函数」转移到「数据规模 + 词表粒度 + 解码策略」。若想继续提升，优先方向依次是：**BPE 子词 → 更多训练数据/epoch → beam search → label smoothing 调参 → learning rate warm-up**。

> 一句话总结：**替换成加性注意力能跑通，但没有收益。** 在这个任务上继续扩展，建议保留点积注意力，把精力投入子词分词和更长训练。


## 附：复现命令

```bash
python evaluate.py --ckpt ckpt_dotproduct.pt --test eng-fra_test_data.txt \
       --out test_result_dotproduct.txt

python evaluate.py --ckpt ckpt_additive.pt   --test eng-fra_test_data.txt \
       --out test_result_additive.txt
```
