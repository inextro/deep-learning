
import os
import torch

from dataset import MNIST
from model import LeNet5, CustomMLP
from matplotlib import pyplot as plt
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


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
        acc: accuracy
    """
    model.train()

    total = 0
    correct = 0
    trn_loss = 0

    for imgs, labels in trn_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad() # 그래디언트 초기화
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward() # 역전파 실행
        optimizer.step() # 파라미터 업데이트

        trn_loss += loss.item()
        _, preds = torch.max(output, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    trn_loss /= len(trn_loader) # 평균 손실 계산
    acc = 100 * correct / total # 정확도 계산

    return trn_loss, acc


def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.eval()

    total = 0
    correct = 0
    tst_loss = 0

    with torch.no_grad():
        for imgs, labels in tst_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            loss = criterion(output, labels)

            tst_loss += loss.item()
            _, preds = torch.max(output, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    tst_loss /= len(tst_loader) # 평균 손실 계산
    acc = 100 * correct / total # 정확도 계산

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    train_data = MNIST(data_dir='./data/train.tar')
    test_data = MNIST(data_dir='./data/test.tar')

    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CrossEntropyLoss()
    epochs = 30

    save_dir = './img'


    # LeNet-5
    print(f'Training LeNet-5 using {device}...')
    lenet = LeNet5().to(device)

    train_losses_lenet = []
    train_accs_lenet = []
    test_losses_lenet = []
    test_accs_lenet = []

    for epoch in range(epochs):
        print(f'Epoch: [{epoch+1}/{epochs}]')

        train_loss_lenet, train_acc_lenet = train(model=lenet, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=SGD(params=lenet.parameters(), lr=0.01, momentum=0.9))
        train_losses_lenet.append(train_loss_lenet)
        train_accs_lenet.append(train_acc_lenet)

        test_loss_lenet, test_acc_lenet = test(model=lenet, tst_loader=test_dataloader, device=device, criterion=criterion)
        test_losses_lenet.append(test_loss_lenet)
        test_accs_lenet.append(test_acc_lenet)
        print(f'Train Loss: {train_loss_lenet: .2f} | Train Acc: {train_acc_lenet: .2f}% | Test Loss: {test_loss_lenet: .2f} | Test Acc: {test_acc_lenet: .2f}%')


        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(range(1, epochs+1), train_losses_lenet)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_accs_lenet)
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.subplot(2, 2, 3)
        plt.plot(epochs, test_losses_lenet)
        plt.title('Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(2, 2, 4)
        plt.plot(epochs, test_accs_lenet)
        plt.title('Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        if not os.path.exists(save_dir): # 경로가 존재하지 않으면 경로 생성
            os.mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, 'LeNet-5.png'))


    # CustomMLP
    print(f'Training CustomMLP using {device}...')
    mlp = CustomMLP().to(device)

    train_losses_mlp = []
    train_accs_mlp = []
    test_losses_mlp = []
    test_accs_mlp = []
    
    for epoch in range(epochs):
        print(f'Epoch: [{epoch+1}/{epochs}]')

        train_loss_mlp, train_acc_mlp = train(model=mlp, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=SGD(params=mlp.parameters(), lr=0.01, momentum=0.9))
        train_losses_mlp.append(train_loss_mlp)
        train_accs_mlp.append(train_acc_mlp)

        test_loss_mlp, test_acc_mlp = test(model=mlp, tst_loader=test_dataloader, device=device, criterion=criterion)
        test_losses_mlp.append(test_loss_mlp)
        test_accs_mlp.append(test_acc_mlp)
        print(f'Train Loss: {train_loss_mlp: .2f} | Train Acc: {train_acc_mlp: .2f}% | Test Loss: {test_loss_mlp: .2f} | Test Acc: {test_acc_mlp: .2f}%')
        print('='*100)


    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), train_losses_mlp)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs_mlp)
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, test_losses_mlp)
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, test_accs_mlp)
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    if not os.path.exists(save_dir): # 경로가 존재하지 않으면 경로 생성
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, 'CustomMLP.png'))


if __name__ == '__main__':
    main()