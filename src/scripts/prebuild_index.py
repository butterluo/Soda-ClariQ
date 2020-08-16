#!/usr/bin/env python
import os
import json
import pandas as pd
from src.scripts.retriever import Indexer


def load_question_bank(data_dir):
    cand_data = []
    questions = pd.read_csv("%s/question_bank.tsv" % data_dir, sep='\t')
    for qid in questions['question_id']:
        q = questions.loc[questions['question_id']==qid, 'question'].tolist()[0]
        cand = {
            "question_id": str(qid),
            "question": str(q)
        }
        cand_data.append(cand)

    return cand_data


def build_mapping(data, index_dir):
    id2question = {}
    for d in data:
        id2question[d['question_id']] = d['question']
    print("total id2question:", len(id2question))
    
    with open("%s/id2question.json" % index_dir, "w") as fw:
        json.dump(id2question, fw, indent=4)

    return id2question


def build(data_dir, index_dir):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    # load data
    data = load_question_bank(data_dir)
    
    # build mappings: <id, question>
    id2question = build_mapping(data, index_dir)

    # build data indexes of questions
    indexer = Indexer(index_dir)
    indexer.build_index(id2question)


if __name__ == "__main__":
    data_dir = "./data"
    index_dir = "./index"
    build(data_dir=data_dir, index_dir=index_dir)
