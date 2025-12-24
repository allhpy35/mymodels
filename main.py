import argparse
import os



# =========================
# Utils import (추후 구현)
# =========================
# from Utils.data_loader import load_data
# from Utils.trainer import train, test
# from Utils.seed import set_seed

# =========================
# Model import (추후 구현)
# =========================
# from Final_models.lstm import LSTMModel
# from Final_models.transformer import TransformerModel

from Utils.get_cuda import get_cuda


def main():
    print('hello world')
    get_cuda()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')

    args = parser.parse_args()
    main()