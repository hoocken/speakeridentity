# Load a dataset
# How? From which dataset? No need to know, what need to know is the path. From train.py probs
# Convert data to Mel spectrograms, need to research this <-

# Choose t = [140, 180] for amount of frames, enforce all utterances of a batch to be this t
import json
import os
import random
import torch

from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

class GE2E_Dataset(data.Dataset):
    def __init__(self, filepath: str, data_dir: str, n_utterances: int, min_seg_length: int, languages: list):
        self.filepath = filepath
        self.data_dir = data_dir
        self.speakers: dict = self._load_speakers(filepath)
        self.n_utterances = n_utterances
        self.min_seg_length = min_seg_length
        self.languages = languages

        self.infos = {}

        for speaker, data in self.speakers.items():
            for l in languages:
                uttrs = [
                    item['path'] for item in data if item['length'] >= min_seg_length and item['language'] == l 
                ]
            if len(uttrs) >= n_utterances:
                self.infos[speaker + l] = uttrs

    def __len__(self):
        return len(self.infos.keys())
    
    def __getitem__(self, idx):
        idx_k = list(self.infos.keys)[idx]
        items = self.infos[idx_k]

        feat_paths = random.sample(items, self.n_utterances)
        utterances = [
            torch.load(os.path.join(self.data_dir, path)) for path in feat_paths
        ]

        # Cut utterances to be of length min_seg_length
        start = [random.randint(0, len(uttr) - self.min_seg_length) for uttr in utterances]
        cut_utterances = [uttr[i: i + self.min_seg_length] for uttr, i in zip(utterances, start)]
        return cut_utterances
        
    def _load_speakers(self, path):
        items = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                speaker = data['speaker']
                
                if not speaker in items.keys():
                    items[speaker] = []

                items[speaker].append(data)
        
        return items
    
def collate_batch(batch):
    flatten = [sample for s in batch for sample in s]
    return pad_sequence(flatten, batch_first=True, padding_value=0)

def build_loader(filepath, data_dir, n_speakers, n_utterances, min_seg_length, num_workers, language):
    dataset = GE2E_Dataset(filepath, data_dir, n_utterances, min_seg_length, language)
        
    train_set, validation_set = data.random_split(dataset, [len(dataset) - n_speakers, n_speakers])

    train_ld = data.DataLoader(
        train_set,
        batch_size=n_speakers,
        collate_fn=collate_batch,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    validation_ld = data.DataLoader(
        validation_set,
        batch_size=n_speakers,
        collate_fn=collate_batch,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    return train_ld, validation_ld