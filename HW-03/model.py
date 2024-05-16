import torch
import numpy as np
import torch.nn as nn

from torch.nn import Linear, RNN, LSTM


class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=1):
        super().__init__()

        self.emb_size = emb_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 임베딩 벡터 look-up table 초기화
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        
        self.rnn = RNN(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.fc = Linear(in_features=hidden_size, out_features=vocab_size)


    def forward(self, input, hidden):
        embedding = self.embedding(input)
        output, hidden = self.rnn(embedding, hidden)
        output = self.fc(output[:, -1, :])
        # batch_first=True인 경우, output의 shape은 (N, L, H_out)
        # N: batch_size, L: seq_len, H_out: hidden_size

        return output, hidden


    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=1):
        super().__init__()

        self.emb_size = emb_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 임베딩 벡터 look-up table 초기화
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)

        self.lstm = LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.fc = Linear(in_features=hidden_size, out_features=vocab_size)


    def forward(self, input, hidden):
        embedding = self.embedding(input)
        output, hidden = self.lstm(embedding, hidden)
        output = self.fc(output[:, -1, :])
        # batch_first=True인 경우, output의 shape은 (N, L, H_out)
        # N: batch_size, L: seq_len, H_out: hidden_size

        return output, hidden


    def init_hidden(self, batch_size):
        initial_hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size), 
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )

        return initial_hidden