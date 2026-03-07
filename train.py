from argparse import ArgumentParser
from os import cpu_count

import torch
from tqdm import tqdm

from dataset import build_loader
from dvector import D_VECTOR
from ge2e import GE2E
from solver import Solver

def main(config):
    train_ld, validation_ld = build_loader(filepath=config.filepath,
                                        data_dir=config.data_dir,
                                        n_speakers=config.n_speakers,
                                        n_utterances=config.n_utterances,
                                        min_seg_length=config.min_seg_length,
                                        num_workers=config.num_workers,
                                        language=config.language)
    
    solver = Solver(model_dir=config.model_dir,
                    train_ld=train_ld,
                    validation_ld=validation_ld,
                    n_speakers=config.n_speakers,
                    n_utterances=config.n_utterances,
                    decay=config.decay,
                    save=config.save,
                    epochs=config.epochs,
                    lr=config.lr,
                    load_state=config.load_state)
    
    solver.train()

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("filepath", type=str)
    PARSER.add_argument("model_dir", type=str)
    PARSER.add_argument("data_dir", type=str)
    PARSER.add_argument("--n_speakers", type=int, default=16)
    PARSER.add_argument("--n_utterances", type=int, default=20)
    PARSER.add_argument("--min_seg_length", type=int, default=160)
    PARSER.add_argument("--lr", type=float, default=0.01)
    PARSER.add_argument("--epochs", type=int, default=200)
    PARSER.add_argument("--save", type=int, default=10)
    PARSER.add_argument("--decay", type=int, default=10)
    PARSER.add_argument("--num_workers", type=int, default=cpu_count())
    PARSER.add_argument("--language", nargs='*', default=['English(US)', 'Japanese', 'Korean', 'Chinese'])
    PARSER.add_argument("--load_state", type=int)

    config = PARSER.parse_args()
    main(config)