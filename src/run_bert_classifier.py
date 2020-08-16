import os
import sys
import torch
import time
import logging
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.nn import CrossEntropyLoss
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from src.datasets.task_data import TaskData
from src.datasets.bert_processor import BertProcessor
from src.utils.utils import collate_fn
from src.utils.monitor import ModelCheckpoint
from src.utils.trainer import Trainer
from src.utils.predictor import Predictor
from src.utils.metrics import F1Score
from src.models.bert_for_classifier import BertForClassifier

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

def run_train(args):
    # read data
    task_data = TaskData(task="task1")
    processor = BertProcessor(vocab_path="%s/vocab.txt" % args.bert_model_dir, do_lower_case=True)

    train_path = "%s/train.tsv" % args.data_dir
    train_data = task_data.read_data(data_path=train_path, is_train=True, shuffle=True)
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train')
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args. max_seq_len)
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn)

    valid_path = "%s/dev.tsv" % args.data_dir
    valid_data = task_data.read_data(data_path=valid_path, is_train=True, shuffle=False)
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid')
    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.max_seq_len)
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, 
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn)

    # set model
    logging.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForClassifier.from_pretrained(args.resume_path, num_labels=task_data.get_num_labels())
    else:
        model = BertForClassifier.from_pretrained(args.bert_model_dir, num_labels=task_data.get_num_labels())
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.num_epochs)
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    
    model_checkpoint = ModelCheckpoint(checkpoint_dir=args.log_dir, mode=args.mode,
                                       save_best_only=args.save_best)

    # training
    logging.info("======= Running training =======")
    logging.info("Num examples = %d" % len(train_examples))
    logging.info("Num epochs = %d" % args.num_epochs)
    logging.info("Gradient accumulation steps = %d" % args.gradient_accumulation_steps)
    logging.info("Total optimization steps = %d" % t_total)

    trainer = Trainer(model=model, num_epochs=args.num_epochs,
                      criterion=CrossEntropyLoss(), optimizer=optimizer,
                      scheduler=scheduler,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[F1Score(task_type='multiclass', average='micro')],
                      epoch_metrics=[F1Score(task_type='multiclass', average='micro')])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)


def run_test(args):
    task_data = TaskData(task="task1")
    processor = BertProcessor(vocab_path="%s/vocab.txt" % args.bert_model_dir, do_lower_case=True)
    
    test_data = task_data.read_data(data_path=args.test_path, 
                                    is_train=False, shuffle=False)
    test_examples = processor.create_examples(lines=test_data,
                                            example_type='test')
    test_features = processor.create_features(examples=test_examples,
                                            max_seq_len=args.max_seq_len)
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, 
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn)
    # predicting
    logging.info('model predicting....')
    model = BertForClassifier.from_pretrained(args.log_dir, num_labels=task_data.get_num_labels())
    predictor = Predictor(task=task_data.get_task(), model=model)
    result = predictor.predict(data=test_dataloader)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task_data.save_predict(raw_data_path=args.test_path, 
                           result_list=result, 
                           save_dir=args.output_dir)


def main():
    parser = ArgumentParser()

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--test_path", default='', type=str)
    parser.add_argument("--bert_model_dir", default='', type=str)
    parser.add_argument("--log_dir", default='', type=str)
    parser.add_argument("--output_dir", default='', type=str)
    
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    torch.save(args, "%s/train_args.bin" % (args.log_dir))
    logging.info("Training/evaluation parameters %s", args)
    
    if args.do_train:
        run_train(args)
    elif args.do_test:
        run_test(args)
    else:
        raise ValueError("train/test args error!")


if __name__ == '__main__':
    main()
