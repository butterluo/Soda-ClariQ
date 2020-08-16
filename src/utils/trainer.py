import logging
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_


class Trainer(object):
    def __init__(self, model, num_epochs, criterion, optimizer, scheduler,epoch_metrics,
                 batch_metrics, gradient_accumulation_steps=1, grad_clip=1.0,
                 model_checkpoint=None):
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        self.model_checkpoint = model_checkpoint
        self.start_epoch = 1
        self.global_step = 0

        self.num_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device {}".format(self.device))
        logging.info("Num GPU {}".format(self.num_gpu))
        self.model = model.to(self.device)
    
    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self,epoch,best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model":model_save,
                 'epoch':epoch,
                 'best':best}
        return state

    def valid_epoch(self,data):
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids,input_mask)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
        
        self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
        self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
        loss = self.criterion(input=self.outputs, target=self.targets)
        self.result['valid_loss'] = loss.item()
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train_epoch(self,data):
        batch_loss = []
        self.epoch_reset()
        for step,  batch in enumerate(data):
            self.batch_reset()
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = self.model(input_ids, segment_ids,input_mask)
            loss = self.criterion(input=logits,target=label_ids)
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            if self.batch_metrics:
                for metric in self.batch_metrics:
                    metric(logits = logits,target = label_ids)
                    self.info[metric.name()] = metric.value()
            self.info['loss'] = loss.item()
            batch_loss.append(loss.item())
            
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
        # epoch metric
        self.outputs = torch.cat(self.outputs, dim =0).cpu().detach()
        self.targets = torch.cat(self.targets, dim =0).cpu().detach()
        self.result['loss'] = np.mean(batch_loss)
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'{metric.name()}'] = value
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self,train_data,valid_data):
        self.model.zero_grad()
        
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            logging.info(f"Epoch {epoch}/{self.num_epochs}")
            train_log = self.train_epoch(train_data)
            valid_log = self.valid_epoch(valid_data)

            logs = dict(train_log,**valid_log)
            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key,value in logs.items()])
            logging.info(show_info)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch,best=None)
                self.model_checkpoint.bert_epoch_step(state = state, current=None)
