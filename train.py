import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATA_DIR = './data'
LOG_DIR = './logs'
MODEL_DIR = './model'

BATCH_SIZE = 16
SEQ_LENGTH = 64

class TrainLogger(object):
    def __init__(self, file, resume=0):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = resume
        if not resume:
            with open(self.file, 'w') as f:
                f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        with open(self.file, 'a') as f:
            f.write(f'{self.epochs},{loss},{acc}\n')

class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, 256, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def read_batches(T, vocab_size):
    length = T.size(0)
    batch_chars = length // BATCH_SIZE

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):
        X = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
        Y = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i]
                Y[batch_idx, i] = T[batch_chars * batch_idx + start + i + 1]
        yield X, Y

def train(text, epochs=100, save_freq=10, resume=False):
    char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
    vocab_size = len(char_to_idx)
    model = LSTMModel(vocab_size)
    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    T = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    log = TrainLogger('training_log.csv', 0)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        model.train()
        total_loss = 0
        for X, Y in read_batches(T, vocab_size):
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = loss_function(Y_pred.transpose(1, 2), Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(T)
        print(f'Average Loss: {avg_loss}')
        log.add_entry(avg_loss, 0)  # placeholder for accuracy

        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'weights.{epoch + 1}.pt'))
            print(f'Saved checkpoint to weights.{epoch + 1}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on text.')
    parser.add_argument('--input', default='nottingham.txt', help='Text file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='Checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(os.path.join(DATA_DIR, args.input), 'r') as data_file:
        text = data_file.read()
    train(text, args.epochs, args.freq)
