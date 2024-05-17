import torch
import argparse
import numpy as np

from model import CharLSTM
from dataset import Shakespeare



def generate(model, seed_characters, temperature, num_chars):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """
    model.eval()
    hidden = model.init_hidden(1)

    samples = seed_characters
    data = Shakespeare(input_file='./data/shakespeare_train.txt')
    idx_to_char = data.idx_to_char
    char_to_idx = data.char_to_idx
    
    # hidden state 초기화
    for char in seed_characters:
        output, hidden = model(torch.tensor([[char_to_idx[char]]]), hidden)
    
    # 문자 생성
    char = seed_characters[-1]
    for _ in range(num_chars):
        output, hidden = model(torch.tensor([[char_to_idx[char]]]), hidden)

        output = output.squeeze().div(temperature).exp()
        char = idx_to_char[torch.multinomial(output, 1)[0].item()]

        samples += char

    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--temp', type=float, default=0.8)
    parser.add_argument('--num_chars', type=int, default=100)
    args = parser.parse_args()
    
    seed_characters = args.seed
    temperature = args.temp
    num_chars = args.num_chars

    # 모델 초기화
    VOCAB_SIZE = 62
    HIDDEN_SIZE = 128

    model = CharLSTM(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load('./model/lstm.pth'))

    # 텍스트 생성
    generated_text = generate(model=model, seed_characters=seed_characters, temperature=temperature, num_chars=num_chars)
    print(generated_text)