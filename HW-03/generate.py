import torch

from dataset import Shakespeare
from model import CharLSTM


def generate(model, seed_characters, temperature, *args):
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

    

    return samples


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Shakespeare(input_file='./data/shakespears_train.txt')
    model = CharLSTM()