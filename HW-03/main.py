import argparse
import numpy as np
import torch
import torch.nn as nn

from dataset import Shakespeare
from model import CharRNN, CharLSTM
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """
    model.to(device)
    model.train()

    loss = 0

    for input, target in trn_loader:
        input, target = input.to(device), target.to(device)

        batch_size = input.size(0)
        init_hidden = model.init_hidden(batch_size=batch_size).to(device)
        
        optimizer.zero_grad() # 그래디언트 초기화
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward() # 역전파 실행
        optimizer().step() # 파라미터 업데이트

    

    return trn_loss


def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here

    return val_loss


def main(epochs, model_name, batch_size, emb_size):
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    data = Shakespeare(input_file='./data/shakespeare_train.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './img'

    vocab_size = len(data.chars)

    index_list = list(range(len(data)))
    np.random.shuffle(index_list)
    split = int(np.floor(0.8 * len(data)))

    train_sampler = SubsetRandomSampler(indices=index_list[:split])
    valid_sampler = SubsetRandomSampler(indices=index_list[split:])

    train_dataloader = DataLoader(dataset=data, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset=data, batch_size=batch_size, sampler=valid_sampler)

    train_loss = []
    val_loss = []

    # RNN
    if model_name == 'rnn':
        print(f'Training RNN using {device}...')
        
        model = CharRNN(vocab_size=vocab_size, emb_size=emb_size)
        criterion  = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters())
        
        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')

            train_loss.append(train(model=model, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=optimizer))

            val_loss.append(validate(model=model, val_loader=valid_dataloader, device=device, criterion=criterion, optimizer=optimizer))

    
    # LSTM
    if model_name == 'lstm':
        print(f'Training LSTM using {device}...')
        
        model = CharLSTM(vocab_size=vocab_size, emb_size=emb_size)
        criterion = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters())

        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')
            train_loss.append(train(model=model, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=optimizer))

            val_loss.append(validate(model=model, val_loader=valid_dataloader, device=device, criterion=criterion, optimizer=optimizer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-m', '--model_name', type=str, required=True, help='rnn or lstm')
    parser.add_argument('b', '--batch_size', type=int, default=64)
    parser.add_argument('-h', '--emb_size', type=int, default=64)
    args = parser.parse_args()

    epochs = args.epcohs
    model_name = args.model_name
    batch_size = args.batch_size
    emb_size = args.emb_size

    main(epoch=epochs, model_name=model_name, batch_size=batch_size, emb_size=emb_size)