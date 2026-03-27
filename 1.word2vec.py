import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.FloatTensor

# 构建词汇表
sentences = [
"the quick brown fox jumps over the lazy dog",
"the lazy dog sleeps all day",
"the quick brown fox is fast",
"the lazy cat sleeps all night",
"apple banana fruit",
"banana orange fruit",
"orange banana fruit",
"dog cat animal",
"cat monkey animal",
"monkey dog animal"
]
sentence_list = " ".join(sentences).split()
vocab = list(set(sentence_list))
word2idx = {w:i for i, w in enumerate(vocab)}
# print(word2idx)
vocab_size = len(vocab)

# model parameters
c = 2  # window size
batch_size = 16
m = 2 # word embedding dim

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
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, True)

class Word2Vec(nn.Module) :
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.w = nn.Parameter(torch.randn(vocab_size, m).type(dtype))  # W
        self.u = nn.Parameter(torch.randn(m, vocab_size).type(dtype))  # U
    def forward(self, X):
        hidden = torch.mm(X,self.w)
        output = torch.mm(hidden,self.u) # [batch_size,vocab_size ]
        return output
def train() :
    torch.manual_seed(777)
    model = Word2Vec()
    loss_fn = nn.CrossEntropyLoss()
    optim = optimizer.Adam(model.parameters(), lr=1e-3)

    total_batches = len(loader)
    print(f'每个epoch的批次数：{total_batches}')

    for epoch in range(1000):
        epoch_loss = 0
        for i , (batch_x , batch_y) in enumerate(loader) :
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

            if (epoch + 1 ) % 100 == 0:
                avg_loss = epoch_loss / total_batches
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 1000, avg_loss))
    for i, label in enumerate(vocab):
        w,wT = model.parameters()
        x,y = float(w[i][0]),float(w[i][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


if __name__ == '__main__':
    train()




