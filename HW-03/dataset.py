import torch

from torch.utils.data import Dataset


class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        super().__init__()
        
        # input_file이 가지고 있는 고유한 문자 추출
        self.input_file = open(input_file).read()
        self.chars = sorted(set(self.input_file))

        # 토큰 딕셔너리 생성
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        
        # 데이터 청크 생성
        self.MAX_LENGTH = 30
        self.chunks = [self.input_file[i:i+self.MAX_LENGTH] for i in range(0, len(self.input_file), self.MAX_LENGTH)]
        
        # 가장 마지막 청크의 길이가 30인지 확인
        if len(self.chunks[-1]) != self.MAX_LENGTH:
            self.chunks = self.chunks[:-1] # 길이가 30이 아니라면 마지막 청크 버리기

        # input 및 target sequence 초기화
        self.input_seqs = []
        self.target_seqs = []

        for chunk in self.chunks:
            input_seq = [self.char_to_idx[c] for c in chunk[:-1]]
            target_seq = [self.char_to_idx[c] for c in chunk[1:]]
            
            self.input_seqs.append(input_seq)
            self.target_seqs.append(target_seq)


    def __len__(self):
        return len(self.input_file)
        

    def __getitem__(self, idx):
        input = torch.tensor(self.input_seqs[idx])
        target = torch.tensor(self.target_seqs[idx])

        return input, target


if __name__ == '__main__':
    temp = Shakespeare(input_file='./data/shakespeare_train.txt')
    
    # __len__ 구현 확인
    print(f'해당 문서는 총 {len(temp)}개의 문자로 구성되어 있습니다. 고유한 문자는 {len(temp.chars)}개 입니다.')

    # 청크 길이 확인
    assert len(temp[0][0]) == temp.MAX_LENGTH - 1
    assert len(temp[0][1]) == temp.MAX_LENGTH - 1
    
    # __getitem__ 구현 확인; 첫번째 데이터 샘플 확인
    print(f'첫번째 입력 청크: {temp[0][0]}')
    print(f'첫번째 타겟 청크: {temp[0][1]}')