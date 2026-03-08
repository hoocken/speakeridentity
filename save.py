from argparse import ArgumentParser

import torch

from dvector import D_VECTOR


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument("load_state", type=str)
    PARSER.add_argument("save", type=str)
    config = PARSER.parse_args()

    dvector = D_VECTOR(dim_input=80)
    checkpoint = torch.load(config.load_state)
    dvector.load_state_dict(checkpoint['dvector_state_dict'])

    torch.save(dvector, config.save)