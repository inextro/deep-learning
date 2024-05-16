import torch
import numpy as np
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()

        # 임베딩 벡터 look-up table 초기화
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        
        self.rnn = nn.RNN(input_size=self.vocab_size, hidden_size=self.emb_size, batch_first=True)
        self.fc = nn.Linear(in_features=self.emb_size, out_features=self.vocab_size)

    def forward(self, input, hidden):
        embedding = self.embedding(input)
        output, hidden = self.rnn(input=embedding, hidden=hidden)
        output = self.fc(output[:, -1, :])
        # batch_first=True인 경우, output의 shape은 (N, L, H_out)
        # N: batch_size, L: seq_len, H_out: hidden_size
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(batch_size, self.emb_size)
        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)

        self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.emb_size, batch_first=True)
        self.fc = nn.Linear(in_features=self.emb_size, out_features=self.vocab_size)

    def forward(self, input, hidden):
        embedding = self.embedding(input)
        output, hidden = self.lstm(input=embedding, hidden=hidden)
        output = self.fc(output[:, -1, :])
        # batch_first=True인 경우, output의 shape은 (N, L, H_out)
        # N: batch_size, L: seq_len, H_out: hidden_size
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(batch_size, self.emb_size)
        return initial_hidden