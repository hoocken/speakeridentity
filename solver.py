from collections import deque
from datetime import datetime

import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dataset import build_loader
from dvector import D_VECTOR
from ge2e import GE2E


class Solver():
    def __init__(self, model_dir, train_ld, validation_ld, lr, n_speakers, n_utterances, decay, epochs, save):
        self.model_dir = model_dir
        self.train_ld = train_ld
        self.validation_ld = validation_ld
        self.lr = lr
        self.n_speakers = n_speakers
        self.n_utterances = n_utterances

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dvector = D_VECTOR(dim_input=80).to(self.device)
        self.criteria = GE2E().to(self.device)
        self.optimizer = torch.optim.Adam(list(self.dvector.parameters()) + list(self.criteria.parameters()), lr=0.01)
        self.scheduler = StepLR(self.optimizer, decay, gamma=0.5) # LR decay

        self.epochs = epochs
        self.save = save

        self.checkpoints = model_dir + '/checkpoints/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(model_dir + '/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        

    @torch.no_grad 
    def validate(self):
        running_valid_loss = deque(maxlen=100)
        for item in tqdm(self.validation_ld, desc='Validating'):
            batch = item.to(self.device)
            embeddings = self.dvector(batch).view(self.n_speakers, self.n_utterances, -1) # (N, M, D)
            loss = self.criteria(embeddings)
            running_valid_loss.append(loss.item())

        return sum(running_valid_loss)/len(running_valid_loss)


    def training(self):
        running_train_loss = deque(maxlen=100)
        
        for item in tqdm(self.train_ld, desc='Training'):
            batch = item.to(self.device)
            embeddings = self.dvector(batch).view(self.n_speakers, self.n_utterances, -1) # (N, M, D)
            loss = self.criteria(embeddings)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step() 

            running_train_loss.append(loss.item())

        return sum(running_train_loss)/len(running_train_loss)

    def train(self):
        for i in range(self.epochs):
            avg_train_loss = self.training()
            self.writer.add_scalar('train/loss', avg_train_loss)
            print(f'[TRAINING: {i + 1} / {self.epochs}] loss = {avg_train_loss}')

            avg_valid_loss = self.validate()
            self.writer.add_scalar('eval/loss', avg_valid_loss)
            print(f'[EVAL: {i + 1} / {self.epochs}] loss = {avg_valid_loss}')

            if i % self.save == 0:
                checkpoint = self.checkpoints + f'/dvector-epoch{i}.pt'
                self.dvector.cpu()
                torch.save(self.dvector, checkpoint)
                self.dvector.to(self.device)

