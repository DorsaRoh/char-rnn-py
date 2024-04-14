# char-rnn-py

Multi-layer recurrent neural networks (RNN, LSTM) for text with PyTorch. Inspired by [Andrej Karpathy](https://github.com/karpathy/char-rnn) and [Eric Zhang](https://github.com/ekzhang/char-rnn-keras/tree/master/data).

## Requirements

Written in Python 3, requires [PyTorch](https://pytorch.org/)

## Input

Place all input data in the [`data/`](./data) directory. Sample training input is provided.

## Usage

To train the model with default settings:
```bash
python train.py --input compositions.txt
```
* compositions is a music notation dataset. Replace and/or add any data you like!

To sample the model at epoch 100:
```bash
python sample.py 100
```

Training loss/accuracy is stored in `logs/training_log.csv`. Model results, including intermediate model weights during training, are stored in the `model` directory. These are also used by `sample.py` for sampling.