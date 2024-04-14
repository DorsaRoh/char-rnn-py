import os

import torch
import torch.nn as nn

MODEL_DIR = './model' 

def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'weights.{}.pt'.format(epoch)))

def load_weights(epoch, model):
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'weights.{}.pt'.format(epoch))))

# build model
class LSTMModel(nn.Module):
    def __init__(self, batch_size, seq_len, vocab_size):
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(512 if i == 0 else 256, 256, batch_first=True)
            for i in range(3)
        ])
        self.dropout = nn.Dropout(0.2)
        # TimeDistributed in pytorch can be achieved using a linear layer since pytorch applies it across time dimension
        self.dense = nn.Linear(256, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    model = LSTMModel(16, 64, 50)
    print(model)