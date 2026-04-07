import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
from sympy.abc import alpha

# 防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 分词
with open("./data/sanguo.txt", 'r',encoding='utf-8') as f:
    lines = []
    for line in f:
        temp = jieba.lcut(line)
        words = []
        for i in temp:
            i = re.sub("[\s+\.\!\/_,$%^*(+\"\'""《》]+|[+——！，。？、~@#￥%……&*（）：；']+","", i)
            if len(i) > 0:
                words.append(i)
        if len(words) > 0:
            lines.append(words)
    print(lines[0:5])


# 模型训练
model = Word2Vec(sentences=lines, vector_size=20, window=2, min_count=3,epochs = 7,negative=10,sg = 1)
print(f"孔明的词向量:\n{model.wv.get_vector('孔明')}")
print(f'\n 和孔明相关性最高的20个词语：{model.wv.most_similar("孔明", topn=20)}')

# 可视化
rawWordVec = []
word2idx ={}
for i,w in enumerate(model.wv.index_to_key):
    rawWordVec.append(model.wv.get_vector(w)) # 词向量
    word2idx[w] = i #{词语:序号}

rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)

# 绘制几个特殊单词的向量
words = ['孙权', '关羽', '张飞', '刘备', '曹操', '周瑜', '小乔', '司马懿','孙尚香']
for w in words:
    if w in word2idx:
        idx = word2idx[w]
        xy = X_reduced[idx]
        plt.plot(xy[0], xy[1], 'o',alpha = 1, color = 'orange',markersize = 10)
        plt.text(xy[0], xy[1], w, alpha = 1, fontsize = 10,color = 'r')

plt.show()




