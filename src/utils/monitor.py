import json
import logging
import numpy as np
from pathlib import Path
import torch


class ModelCheckpoint(object):
    def __init__(self, checkpoint_dir,
                 mode='min',
                 epoch_freq=1,
                 best = None,
                 save_best_only = True):
        if isinstance(checkpoint_dir,Path):
            checkpoint_dir = checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.is_dir()
        checkpoint_dir.mkdir(exist_ok=True)
        self.base_path = checkpoint_dir
        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        
        if best:
            self.best = best

        if save_best_only:
            self.model_name = f"BEST_MODEL.pth"

    def epoch_step(self, state,current):
        '''
        :param state: 
        :param current:
        :return:
        '''
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                logging.info(f"\nEpoch {state['epoch']}: improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                best_path = self.base_path/ self.model_name
                torch.save(state, str(best_path))

        else:
            filename = self.base_path / f"epoch_{state['epoch']}_model.bin"
            if state['epoch'] % self.epoch_freq == 0:
                logging.info(f"\nEpoch {state['epoch']}: save model to disk.")
                torch.save(state, str(filename))

    def bert_epoch_step(self, state,current):
        model_to_save = state['model']
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                logging.info(f"\nEpoch {state['epoch']}: improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                model_to_save.save_pretrained(str(self.base_path))
                output_config_file = self.base_path / 'config.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state,self.base_path / 'checkpoint_info.bin')

        else:
            if state['epoch'] % self.epoch_freq == 0:
                save_path = self.base_path / f"checkpoint-epoch-{state['epoch']}"
                save_path.mkdir(exist_ok=True)
                logging.info(f"\nEpoch {state['epoch']}: save model to disk.")
                model_to_save.save_pretrained(save_path)
                output_config_file = save_path / 'config.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state, save_path / 'checkpoint_info.bin')
