
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self, regularization=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

        if regularization:
            self.dropout = nn.Dropout() # droput 적용
        
        else:
            self.dropout = nn.Identity() # dropout 적용하지 않음

    def forward(self, img):
        output = F.relu(self.conv1(img))
        output = F.max_pool2d(output, kernel_size=2)
        output = F.relu(self.conv2(output))
        output = F.max_pool2d(output, kernel_size=2)
        output = output.view(output.size(0), -1)
        output = self.dropout(output) # dropout 적용
        output = F.relu(self.fc1(output))
        output = self.dropout(output) # dropout 적용
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=32*32, out_features=56)
        self.fc2 = nn.Linear(in_features=56, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, img):
        img = img.view(img.size(0), -1)
        output = F.relu(self.fc1(img))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output