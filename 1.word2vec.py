import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data
import matplotlib.pyplot as plt
import re
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.FloatTensor

# 读取并预处理文本
def load_and_preprocess_text(file_path, min_freq=5):
    """
    从文件加载文本并进行预处理
    :param file_path: 文本文件路径
    :param min_freq: 最小词频，低于此频率的词将被忽略
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_freq]

    print(f"总词数：{len(words)}")
    print(f"词汇表大小（频率>={min_freq}, 最大={max_vocab_size}）: {len(vocab)}")
    
    word2idx = {w: i for i, w in enumerate(vocab)}
    sentence_list = [word for word in words if word in word2idx]
    
    return sentence_list, vocab, word2idx

sentence_list, vocab, word2idx = load_and_preprocess_text(
    '/Users/yzx/PycharmProjects/DL_course/data/Wuthering Heights.txt', 
    min_freq=10,
    max_vocab_size=500  # 限制为 500 个词
)
vocab_size = len(vocab)

# model parameters
c = 2  # 窗口
batch_size = 128  # 增大批次大小以适应更大的数据集
m = 50  # 增加词向量维度以获得更好的表示

skip_grams = []
for idx in range(c,len(sentence_list)-c) :
    center = word2idx[sentence_list[idx]]
    context_idx = list(range(idx-c,idx)) + list(range(idx+1,idx+c+1))
    context = [word2idx[sentence_list[i]] for i in context_idx]
    for w in context :
        skip_grams.append([center,w])

def make_data(skip_grams) :
    input_data= []
    output_data = []
    for a,b in skip_grams:
        input_data.append((np.eye(vocab_size)[a]))
        output_data.append(b)
    return  input_data,output_data

input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(input_data), torch.Tensor(output_data)
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, True)

class Word2Vec(nn.Module) :
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.w = nn.Parameter(torch.randn(vocab_size, m).type(dtype))  # W
        self.u = nn.Parameter(torch.randn(m, vocab_size).type(dtype))  # U
    def forward(self, X):
        hidden = torch.mm(X,self.w) #隐藏层
        output = torch.mm(hidden,self.u) # [batch_size,vocab_size ]
        return output
def train() :
    model = Word2Vec()
    loss_fn = nn.CrossEntropyLoss()
    optim = optimizer.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2000):
        for i , (batch_x , batch_y) in enumerate(loader) :
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            if (epoch + 1 ) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 2000, loss.item()))
            optim.zero_grad()
            loss.backward()
            optim.step()

    for i, label in enumerate(vocab):
        w,wT = model.parameters ()
        x,y = float(w[i][0]),float(w[i][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


if __name__ == '__main__':
    train()




