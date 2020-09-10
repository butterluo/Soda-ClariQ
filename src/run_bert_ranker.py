import pandas as pd
import logging
import os
import sys
import torch
import random
import argparse
import numpy as np
from src.models import transformer_ranker
from src.datasets.dataset import SimpleDataset, QueryDocumentDataLoader
from src.datasets.data_collator import DefaultDataCollator
from src.utils import negative_sampling
from src.utils import results_analyses_tools
from src.utils import utils
from src.utils import bm25
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import DataLoader

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

def load_data(data_dir):
    train = pd.read_csv("%s/train.tsv" % data_dir, sep="\t")
    train = train[["initial_request", "question"]]
    train.columns = ["query", "clarifying_question"]
    train = train[~train["clarifying_question"].isnull()]

    dev = pd.read_csv("%s/dev.tsv" % data_dir, sep="\t")
    dev = dev[["initial_request", "question"]]
    dev.columns = ["query", "clarifying_question"]
    dev = dev[~dev["clarifying_question"].isnull()]
        
    # We will sample negative samples for training using the question bank
    question_bank = pd.read_csv("%s/question_bank.tsv" % data_dir, sep="\t")
    return train, dev, question_bank
        

def run_train(args):
    train, dev, question_bank = load_data(data_dir=args.data_dir)

    #use an almost balanced amount of positive and negative samples during training
    average_relevant_per_query = train.groupby("query").count().mean().values[0]
    ns_train = negative_sampling.RandomNegativeSampler(
        list(question_bank["question"].values[1:]), int(average_relevant_per_query))
    ns_val = negative_sampling.RandomNegativeSampler(
        list(question_bank["question"].values[1:]), int(average_relevant_per_query))
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cache_dir = "%s/cache_data" % args.log_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt" % args.bert_model_dir)
    dataloader = QueryDocumentDataLoader(
        train_df=train, val_df=dev, test_df=dev,
        tokenizer=tokenizer, negative_sampler_train=ns_train,
        negative_sampler_val=ns_val, task_type='classification',
        train_batch_size=args.batch_size, val_batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        sample_data=-1, cache_path=cache_dir)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()
    
    #use BERT (any model that has SequenceClassification class from HuggingFace would work here)
    model = BertForSequenceClassification.from_pretrained(args.bert_model_dir)
    ranker = transformer_ranker.TransformerRanker(
        model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        num_ns_eval=int(average_relevant_per_query), task_type="classification", tokenizer=tokenizer,
        validate_every_epochs=1, num_validation_instances=-1,
        num_epochs=args.num_epochs, lr=args.lr, sacred_ex=None)
    
    tokenizer.save_pretrained(args.log_dir)
    ranker.fit(log_dir=args.log_dir)  


def get_best_q(query, question_bank, top_n=5):
    bm25_qids = bm25.get_top_n(query, question_bank, top_n=top_n)
    best_qid = bm25_qids[0]
    return best_qid


def run_test(args):
    data = pd.read_csv(args.test_path, sep='\t')
    question_bank = pd.read_csv("%s/question_bank.tsv" % args.data_dir, sep="\t")
    all_documents = list(question_bank["question"].values[1:])
    examples = []
    for tid in data['topic_id'].unique():
        query = data.loc[data['topic_id']==tid, 'initial_request'].tolist()[0]
        for doc in all_documents:
            examples.append((query, doc))
    
    tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt" % args.log_dir)
    batch_encoding = tokenizer.batch_encode_plus(
        examples, max_length=args.max_seq_len, truncation=True, pad_to_max_length=True)
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=0)
        features.append(feature)
    dataset = SimpleDataset(features)
    data_collator = DefaultDataCollator()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator.collate_batch)
    
    # load fine-tuned model
    model = BertForSequenceClassification.from_pretrained(args.log_dir)
    ranker = transformer_ranker.TransformerRanker(
        model=model, train_loader=None, val_loader=None, test_loader=None,
        num_ns_eval=None, task_type="classification", tokenizer=tokenizer,
        validate_every_epochs=1, num_validation_instances=-1,
        num_epochs=args.num_epochs, lr=args.lr, sacred_ex=None)
    _, _, softmax_output = ranker.predict(dataloader)
    softmax_output_by_query = utils.acumulate_list(softmax_output[0], len(all_documents))

    # save output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if "dev" in args.test_path:
        run_file_path = "%s/dev_ranked_q.txt" % args.output_dir
    else:
        run_file_path = "%s/test_ranked_q.txt" % args.output_dir
    all_doc_ids = np.array(question_bank["question_id"].values[1:])
    with open(run_file_path, 'w') as fo:
        for tid_idx, tid in enumerate(data['topic_id'].unique()):
            all_documents_scores = np.array(softmax_output_by_query[tid_idx])
            print("tid:", tid)
            
            top_30_scores_idx = (-all_documents_scores).argsort()[:30]  
            preds_score = list(all_documents_scores[top_30_scores_idx])
            preds = list(all_doc_ids[top_30_scores_idx])
            #print("softmax_score:", preds_score)
            #print("preds:", preds)
            #query = data.loc[data['topic_id']==tid, 'initial_request'].tolist()[0]
            #best_q = get_best_q(query, question_bank)
            #best_qid = random.choice([best_q, "Q00001"])
            
            if preds_score[0] < 0.962:
                best_qid = "Q00001"
                preds = preds[:-1]
                preds.insert(0, best_qid)
            else:
                last_qid = "Q00001"
                preds = preds[:-1]
                preds.append(last_qid)
            for i, qid in enumerate(preds):    
                fo.write('{} 0 {} {} {} BERT-based-v2\n'.format(tid, qid, i, len(preds)-i))
    print("saved results to [%s]" % run_file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--test_path", default='', type=str)
    
    parser.add_argument("--bert_model_dir", default='', type=str)
    parser.add_argument("--log_dir", default='', type=str)
    parser.add_argument("--output_dir", default='', type=str)

    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=5e-7, type=float)
    args = parser.parse_args()

    if args.do_train:
        run_train(args)
    elif args.do_test:
        run_test(args)
    else:
        raise ValueError("do_train or do_test should be set!")
    

if __name__ == "__main__":
    main()
