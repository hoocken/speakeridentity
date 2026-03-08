from collections import deque
from datetime import datetime
import os

import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dataloader import infinite_iterator
from dataset import build_loader
from dvector import D_VECTOR
from ge2e import GE2E
from itertools import count


class Solver():
    def __init__(self, model_dir, train_ld, validation_ld, lr, n_speakers, n_utterances, decay, save, load_state, valid_every, batch_per_valid):
        self.model_dir = model_dir
        self.train_ld = train_ld
        self.validation_ld = validation_ld

        self.train_iter = infinite_iterator(self.train_ld)
        self.valid_iter = infinite_iterator(self.validation_ld)

        self.lr = lr
        self.n_speakers = n_speakers
        self.n_utterances = n_utterances

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dvector = D_VECTOR(dim_input=80).to(self.device)
        self.criteria = GE2E().to(self.device)
        self.optimizer = torch.optim.SGD(list(self.dvector.parameters()) + list(self.criteria.parameters()), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, decay, gamma=0.5) # LR decay

        self.valid_every = valid_every
        self.start = 0
        self.batch_per_valid = batch_per_valid

        if load_state:
            checkpoint = torch.load(load_state, weights_only=True)
            self.dvector.load_state_dict(checkpoint['dvector_state_dict'])
            self.criteria.load_state_dict(checkpoint['criteria_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.scheduler.last_epoch = self.epoch

        self.save = save

        self.checkpoints = model_dir + '/checkpoints/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            os.makedirs(self.checkpoints)
        except FileExistsError:
            print(f'{self.checkpoints} already exists!')

        self.writer = SummaryWriter(model_dir + '/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        

    @torch.no_grad 
    def validate(self):
        running_valid_loss = deque(maxlen=100)
        for _ in range(self.batch_per_valid):
            batch = next(self.valid_iter).to(self.device)
            embeddings = self.dvector(batch).view(self.n_speakers, self.n_utterances, -1) # (N, M, D)
            loss = self.criteria(embeddings)
            running_valid_loss.append(loss.item())

        return sum(running_valid_loss)/len(running_valid_loss)

    def train(self):
        pbar = tqdm(total=self.valid_every, initial=self.epoch, ncols=0, desc="Train")
        running_train_loss = deque(maxlen=100)
        running_grad_norm = deque(maxlen=100)
        
        for i in count(start=self.epoch):
            batch = next(self.train_iter).to(self.device)
            embeddings = self.dvector(batch).view(self.n_speakers, self.n_utterances, -1) # (N, M, D)
            loss = self.criteria(embeddings)

            self.optimizer.zero_grad()
            loss.backward()

            self.dvector.embedding.weight.grad *= 0.5
            self.dvector.embedding.bias.grad *= 0.5
            self.criteria.w.grad *= 0.01
            self.criteria.b.grad *= 0.01

            grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.dvector.parameters()) + list(self.criteria.parameters()),
                    max_norm=3,
                    norm_type=2.0,
                )

            self.optimizer.step()
            self.scheduler.step() 

            running_train_loss.append(loss.item())
            running_grad_norm.append(grad_norm.item())

            avg_train_loss, avg_grad_norm = sum(running_train_loss)/len(running_train_loss), sum(running_grad_norm)/len(running_grad_norm)
            
            pbar.update(1)
            pbar.set_postfix(loss=avg_train_loss, grad_norm=avg_grad_norm)

            if i % self.valid_every == 0: 
                self.writer.add_scalar('train/loss', avg_train_loss)
                self.writer.add_scalar('train/grad', avg_grad_norm)

                avg_valid_loss = self.validate()
                self.writer.add_scalar('eval/loss', avg_valid_loss)
                tqdm.write(f'[EVAL: {i + 1}] loss = {avg_valid_loss}')

            if i % self.save == 0:
                checkpoint = self.checkpoints + f'/dvector-epoch{i}.pt'
                self.dvector.cpu()
                torch.save({
                    'epoch': i,
                    'dvector_state_dict': self.dvector.state_dict(),
                    'criteria_state_dict': self.criteria.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_valid_loss,
                    }, checkpoint)
                self.dvector.to(self.device)

