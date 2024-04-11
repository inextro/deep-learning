import torch
import tarfile
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

        # transform 정의
        self.transform = transforms.Compose([
            transforms.Resize(size=(32, 32)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
        ])

        # 이미지 이름 목록 생성
        self.members = []
        with tarfile.open(data_dir, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.png'):
                    self.members.append(member.name)

    def __len__(self):
        return len(self.members) # 전체 이미지의 개수 반환

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 이미 파일 선택
        with tarfile.open(self.data_dir, 'r') as tar:
            member = tar.getmember(self.members[idx])
            img = tar.extractfile(member)
            img = Image.open(img)
            img = self.transform(img) # transform 적용

        # 이미지 파일 이름에서 라벨 추출
        label = int(member.name.split('_')[1].split('.')[0])

        return img, label

if __name__ == '__main__':
    # __len__ 구현 확인
    train_data = MNIST(data_dir='./data/train.tar')
    test_data = MNIST(data_dir='./data/test.tar')

    assert len(train_data) == 60000 # 학습 데이터 6만장
    assert len(test_data) == 10000 # 검증 데이터 1만장

    # __getitem__ 구현 확인; 최초 5개의 이미지에 대해서 검증 수행
    for i in range(5):
        img, label = train_data[i]
        print(f'[{i+1}/5] Label of Current Image is', label)
        plt.imshow(np.transpose(img, (1, 2, 0))) # (C, H, W) -> (H, W, C)로 변경
        plt.show()