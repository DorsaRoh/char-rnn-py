import argparse
import json
import os
import torch
import torch.nn as nn

DATA_DIR = './data'
DATASET_DIR = 'data/compositions.txt'
MODEL_DIR = './model'
JSON_DIR = 'model/char_to_idx.json'


# generate char_to_idx.json
with open(DATASET_DIR, 'r', encoding='utf-8') as file:
    data = file.read()
unique_chars = sorted(set(data))
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
with open(JSON_DIR, 'w', encoding='utf-8') as json_file:
    json.dump(char_to_idx, json_file)

class SampleModel(nn.Module):
    def __init__(self, vocab_size):
        super(SampleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, 256, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden

    def init_hidden(self):
        return (torch.zeros(3, 1, 256), torch.zeros(3, 1, 256))

def sample(epoch, header, num_chars, model_path):
    with open(os.path.join(MODEL_DIR, 'char_to_idx.json'), 'r') as f:
        char_to_idx = json.load(f)
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    model = SampleModel(vocab_size)
    model.load_state_dict(torch.load(os.path.join(model_path, f'weights.{epoch}.pt')))
    model.eval()

    hidden = model.init_hidden()
    sampled = [char_to_idx[c] for c in header] if header else [torch.randint(vocab_size, (1,)).item()]
    for c in header[:-1]:
        batch = torch.tensor([[char_to_idx[c]]], dtype=torch.long)
        _, hidden = model(batch, hidden)

    for i in range(num_chars):
        batch = torch.tensor([[sampled[-1]]], dtype=torch.long)
        with torch.no_grad():
            output, hidden = model(batch, hidden)
        probabilities = torch.softmax(output[0, -1], dim=0).data
        sample = torch.multinomial(probabilities, 1)[0]
        sampled.append(sample.item())

    return ''.join(idx_to_char[c] for c in sampled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample some text from the trained model.')
    parser.add_argument('epoch', type=int, help='epoch checkpoint to sample from')
    parser.add_argument('--seed', default='', help='initial seed for the generated text')
    parser.add_argument('--len', type=int, default=420, help='number of characters to sample')
    args = parser.parse_args()

    # ensure DATA_DIR and MODEL_DIR exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(sample(args.epoch, args.seed, args.len, MODEL_DIR))
