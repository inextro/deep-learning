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

    samples = None
    

    return samples


if __name__ == '__main__':
    pass