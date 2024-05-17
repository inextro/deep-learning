import os
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

    trn_loss = 0

    for input, target in trn_loader:
        input, target = input.to(device), target.to(device)

        batch_size = input.size(0)
        
        if model_name == 'rnn':
            hidden = model.init_hidden(batch_size=batch_size).to(device)
        elif model_name == 'lstm':
            hidden = model.init_hidden(batch_size=batch_size)
            hidden = (hidden[0].to(device), hidden[1].to(device))
        
        optimizer.zero_grad()
        output, hidden = model(input, hidden)

        loss = criterion(output.view(-1, model.vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

    trn_loss = trn_loss / len(trn_loader) # 배치별 평균 손실 함수값 계산

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
    model.to(device)
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)

            batch_size = input.size(0)
            
            if model_name == 'rnn':
                hidden = model.init_hidden(batch_size=batch_size).to(device)
            elif model_name == 'lstm':
                hidden = model.init_hidden(batch_size=batch_size)
                hidden = (hidden[0].to(device), hidden[1].to(device))

            output, hidden = model(input, hidden)
            loss = criterion(output.view(-1, model.vocab_size), target.view(-1))

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    return val_loss


def main(epochs, model_name, batch_size, hidden_size):
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
    plot_save_dir = './img'
    model_save_dir = './model'

    vocab_size = len(data.chars)

    index_list = list(range(len(data)))
    np.random.shuffle(index_list)
    split = int(np.floor(0.8 * len(data)))

    train_sampler = SubsetRandomSampler(indices=index_list[:split])
    valid_sampler = SubsetRandomSampler(indices=index_list[split:])

    train_dataloader = DataLoader(dataset=data, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset=data, batch_size=batch_size, sampler=valid_sampler)

    train_losses = []
    val_losses = []

    # RNN
    if model_name == 'rnn':
        print(f'Training RNN using {device}...')
        
        model = CharRNN(vocab_size=vocab_size, hidden_size=hidden_size)
        criterion  = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters())
        
        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')

            train_loss = train(model=model, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=optimizer)
            train_losses.append(train_loss)

            val_loss = validate(model=model, val_loader=valid_dataloader, device=device, criterion=criterion)
            val_losses.append(val_loss)

            print(f'Train Loss: {train_loss}', '\t', f'Valid Loss: {val_loss}')
    
    # LSTM
    if model_name == 'lstm':
        print(f'Training LSTM using {device}...')
        
        model = CharLSTM(vocab_size=vocab_size, hidden_size=hidden_size)
        criterion = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters())

        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')

            train_loss = train(model=model, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=optimizer)
            train_losses.append(train_loss)

            val_loss = validate(model=model, val_loader=valid_dataloader, device=device, criterion=criterion)
            val_losses.append(val_loss)

            print(f'Train Loss: {train_loss}', '\t', f'Valid Loss: {val_loss}')

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')

    if model_name == 'rnn':
        plt.title('[CharRNN] Cross Entropy Loss by Epochs')
    elif model_name == 'lstm':
        plt.title('[CharLSTM] Cross Entropy Loss by Epochs')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save
    if not os.path.exists(plot_save_dir): # 경로가 존재하지 않으면 경로 생성
        os.mkdir(plot_save_dir)
    plt.savefig(os.path.join(plot_save_dir, f'{model_name}.png'))

    if not os.path.exists(model_save_dir): # 경로가 존재하지 않으면 경로 생성
        os.mkdir(model_save_dir)
    torch.save(model.state_dict(), os.path.join(model_save_dir, f'{model_name}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--model_name', type=str, required=True, help='rnn or lstm')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    args = parser.parse_args()

    epochs = args.epochs
    model_name = args.model_name
    batch_size = args.batch_size
    hidden_size = args.hidden_size

    main(epochs=epochs, model_name=model_name, batch_size=batch_size, hidden_size=hidden_size)