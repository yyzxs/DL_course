import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader,TensorDataset
import time


def load_text():
    f = open("data/sanguo.txt", 'r', encoding='utf-8')
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
    return lines

def build_vocab(lines):
    vocab = set()
    for line in lines:
        for word in line:
            vocab.add(word)
    
    vocab = list(vocab)
    word2idx = {w:i for i, w in enumerate(vocab)}
    idx2word = {i:w for i, w in enumerate(vocab)}
    return vocab, word2idx, idx2word

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

def create_skip_grams(lines, word2idx, window_size=2):
    skip_grams = []
    for line in lines:
        for idx in range(len(line)):
            if line[idx] not in word2idx:
                continue
            center = word2idx[line[idx]]
            start = max(0, idx - window_size)
            end = min(len(line), idx + window_size + 1)
            for context_idx in range(start, end):
                if context_idx != idx and line[context_idx] in word2idx:
                    context = word2idx[line[context_idx]]
                    skip_grams.append([center, context])
    return skip_grams

def make_dataset(skip_grams, vocab_size):
    input_data = []
    output_data = []
    for center, context in skip_grams:
        input_data.append(center)
        output_data.append(context)
    input_data = torch.LongTensor(input_data)
    output_data = torch.LongTensor(output_data)
    dataset = TensorDataset(input_data, output_data)
    return dataset

def train(dataset, vocab_size):
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = Word2Vec(vocab_size, 100)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        start_time = time.time()
        
        for x, y in train_loader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            _, predicted = torch.max(y_pred, 1)
            total_correct += (predicted == y).sum().item()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s')
    
    return model

if __name__ == '__main__':
     lines = load_text()
     vocab, word2idx, idx2word = build_vocab(lines)
     print(f'词汇表大小：{len(vocab)}')
     
     skip_grams = create_skip_grams(lines, word2idx)
     print(f'Skip-gram 数量：{len(skip_grams)}')
     
     dataset = make_dataset(skip_grams, len(vocab))
     model = train(dataset, len(vocab))
     
     w = model.embedding.weight.data.numpy()
     pca = PCA(n_components=2)
     w_pca = pca.fit_transform(w[:100])
     
     plt.figure(figsize=(10, 8))
     for i in range(min(100, len(vocab))):
         plt.scatter(w_pca[i, 0], w_pca[i, 1])
         plt.annotate(vocab[i], xy=(w_pca[i, 0], w_pca[i, 1]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', fontsize=8)
     plt.title('Word2Vec Visualization (PCA)')
     plt.show()
