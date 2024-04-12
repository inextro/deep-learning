
import os
import torch
import argparse

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


def main(epochs, model_name, regularization):
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
    save_dir = './img'

    # LeNet-5
    if model_name == 'lenet':
        print(f'Training LeNet-5 using {device}...')
        print(f'Training LeNet-5 with regularization: {regularization}')
        
        if regularization: # regularization
            lenet = LeNet5(regularization=regularization).to(device)
        else:
            lenet = LeNet5().to(device)

        train_losses_lenet = []
        train_accs_lenet = []
        test_losses_lenet = []
        test_accs_lenet = []

        best_loss_lenet = None
        best_acc_lenet = None

        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')

            if regularization:
                train_loss_lenet, train_acc_lenet = train(model=lenet, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=SGD(params=lenet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01))
                test_loss_lenet, test_acc_lenet = test(model=lenet, tst_loader=test_dataloader, device=device, criterion=criterion)

            else:
                train_loss_lenet, train_acc_lenet = train(model=lenet, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=SGD(params=lenet.parameters(), lr=0.01, momentum=0.9))
                test_loss_lenet, test_acc_lenet = test(model=lenet, tst_loader=test_dataloader, device=device, criterion=criterion)

            # 현재 최고 기록 저장
            if best_loss_lenet is None:
                best_loss_lenet = test_loss_lenet
            
            elif test_loss_lenet < best_loss_lenet:
                best_loss_lenet = test_loss_lenet
                print(f'New Best Loss Record at Epoch {epoch}! | Best Loss: {best_loss_lenet}')
            
            if best_acc_lenet is None:
                best_acc_lenet = test_acc_lenet
            
            elif best_acc_lenet < test_acc_lenet:
                best_acc_lenet = test_acc_lenet
                print(f'New Best Acc Record at Epoch {epoch}! | Best Acc: {best_acc_lenet}')

            train_losses_lenet.append(train_loss_lenet)
            train_accs_lenet.append(train_acc_lenet)
            test_losses_lenet.append(test_loss_lenet)
            test_accs_lenet.append(test_acc_lenet)

            print(f'Train Loss: {train_loss_lenet: .2f} | Train Acc: {train_acc_lenet: .2f}% | Test Loss: {test_loss_lenet: .2f} | Test Acc: {test_acc_lenet: .2f}%')
            print('='*100)

        plt.figure(figsize=(10, 4))
        plt.suptitle('LeNet-5 Training and Test Metrics')
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), train_losses_lenet, label='Train Loss')
        plt.plot(range(1, epochs+1), test_losses_lenet, label='Test Loss')
        plt.title('Train/Test Loss')
        plt.xlabel('Time (epochs)')
        plt.ylabel('Loss (cross entropy)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), train_accs_lenet, label='Train Accuracy')
        plt.plot(range(1, epochs+1), test_accs_lenet, label='Test Accuracy')
        plt.title('Train/Test Accuracy')
        plt.xlabel('Time (epochs)')
        plt.ylabel('Accuracy')
        plt.legend()


        if not os.path.exists(save_dir): # 경로가 존재하지 않으면 경로 생성
            os.mkdir(save_dir)
        plt.tight_layout()

        if regularization:
            plt.savefig(os.path.join(save_dir, 'LeNet-5_reg.png'))
        else:
            plt.savefig(os.path.join(save_dir, 'LeNet-5.png'))

    elif model_name == 'mlp':
        # CustomMLP
        print(f'Training CustomMLP using {device}...')
        mlp = CustomMLP().to(device)

        train_losses_mlp = []
        train_accs_mlp = []
        test_losses_mlp = []
        test_accs_mlp = []

        best_loss_mlp = None
        best_acc_mlp = None
        
        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')

            train_loss_mlp, train_acc_mlp = train(model=mlp, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=SGD(params=mlp.parameters(), lr=0.01, momentum=0.9))
            test_loss_mlp, test_acc_mlp = test(model=mlp, tst_loader=test_dataloader, device=device, criterion=criterion)

            # 현재 최고 기록 저장
            if best_loss_mlp is None:
                best_loss_mlp = test_loss_mlp
            
            elif test_loss_mlp < best_loss_mlp:
                best_loss_mlp = test_loss_mlp
                print(f'New Best Loss Record at Epoch {epoch}! | Best Loss: {best_loss_mlp}')
            
            if best_acc_mlp is None:
                best_acc_mlp = test_acc_mlp
            
            elif best_acc_mlp < test_acc_mlp:
                best_acc_mlp = test_acc_mlp
                print(f'New Best Acc Record at Epoch {epoch}! | Best Acc: {best_acc_mlp}')

            train_losses_mlp.append(train_loss_mlp)
            train_accs_mlp.append(train_acc_mlp)
            test_losses_mlp.append(test_loss_mlp)
            test_accs_mlp.append(test_acc_mlp)

            print(f'Train Loss: {train_loss_mlp: .2f} | Train Acc: {train_acc_mlp: .2f}% | Test Loss: {test_loss_mlp: .2f} | Test Acc: {test_acc_mlp: .2f}%')
            print('='*100)

        plt.figure(figsize=(10, 4))
        plt.suptitle('CustomMLP Training and Test Metrics')
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), train_losses_mlp, label='Train Loss')
        plt.plot(range(1, epochs+1), test_losses_mlp, label='Test Loss')
        plt.title('Train/Test Loss')
        plt.xlabel('Time (epochs')
        plt.ylabel('Loss (cross entropy)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), train_accs_mlp, label='Train Accuracy')
        plt.plot(range(1, epochs+1), test_accs_mlp, label='Test Accuracy')
        plt.title('Train/Test Accuracy')
        plt.xlabel('Time (epochs)')
        plt.ylabel('Accuracy')
        plt.legend()


        if not os.path.exists(save_dir): # 경로가 존재하지 않으면 경로 생성
            os.mkdir(save_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'CustomMLP.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-m', '--model', type=str, required=True, help='lenet or mlp')
    parser.add_argument('-r', '--regularization', type=bool, default=False)
    args = parser.parse_args()

    epochs = args.epochs
    model_name = args.model
    regularization = args.regularization
    
    main(epochs=epochs, model_name=model_name, regularization=regularization)