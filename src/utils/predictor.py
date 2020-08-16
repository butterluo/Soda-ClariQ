#encoding:utf-8
import torch
import numpy as np
import logging


class Predictor(object):
    def __init__(self, task, model):
        self.task = task
        assert self.task in ["task1", "task2"]
        self.num_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device {}".format(self.device))
        logging.info("Num GPU {}".format(self.num_gpu))
        self.model = model.to(self.device)

    def predict(self, data):
        preds = []
        if self.task == "task1":
            for step, batch in enumerate(data):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    input_ids, input_mask, segment_ids, label_ids = batch
                    logits = self.model(input_ids, segment_ids, input_mask)
                    logits = logits.softmax(-1)
                    y_prob = logits.detach().cpu().numpy()
                    y_pred = np.argmax(y_prob, 1)
                    for pred in y_pred:
                        preds.append(pred)
        else:
            for step, batch in enumerate(data):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    input_ids, input_mask, segment_ids, label_ids = batch
                    logits = self.model(input_ids, segment_ids, input_mask)
                    #logits = logits.sigmoid()
                    #y_prob = logits.detach().cpu().numpy()
                    #y_pred = (y_prob > 0.5).astype(int)
                    logits = logits.softmax(-1)
                    y_prob = logits.detach().cpu().numpy()
                    y_pred = np.argmax(y_prob, 1)
                    for pred in y_pred:
                        preds.append(pred)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return preds
