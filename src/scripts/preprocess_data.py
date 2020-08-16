import pandas as pd
import pickle
import json
import re
import functools
from os import path
from statistics import mean
from src.scripts.retriever import Queryer


replacement = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "i would",
    "i'll" : "i will",
    "i'm" : "i am",
    "i’m": "i am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll": "it will",
    "i've" : "i have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " we will",
    "tryin'": "trying",
}

def evaluate_document_relevance(eval_dict, facet_to_topic_dict, run_dict):
    performance_dict = {}
    for metric in eval_dict:
        performance_dict[metric] = {}
        get_document_relevance_for_metric(eval_dict, facet_to_topic_dict, metric, performance_dict,
                                          run_dict)
    # compute the mean performance per metric and print
    result_dict = {}
    for metric in performance_dict:
        result_dict[metric] = mean(performance_dict[metric][k] for k in performance_dict[metric])
    return result_dict


def get_document_relevance_for_metric(eval_dict, facet_to_topic_dict, metric, performance_dict, run_dict):
    for facet_id in eval_dict[metric]:
        try:
            selected_q = get_selected_question(facet_id, facet_to_topic_dict, run_dict, multi_turn=False)
            try:
                performance_dict[metric][facet_id] = eval_dict[metric][facet_id][selected_q]['with_answer']
            except KeyError:  # if question is not among candidate question, we consider it equal to minimum performance.
                performance_dict[metric][facet_id] = eval_dict[metric][facet_id]['MIN']['with_answer']
        except KeyError:  # if there is no prediction provided for a facet, we consider performance 0.
            performance_dict[metric][facet_id] = 0.


def get_selected_question(facet_id, facet_to_topic_dict, run_dict, multi_turn=False):
    if multi_turn:
        selected_q = run_dict[facet_id]
    else:
        selected_q = run_dict[facet_to_topic_dict[facet_id]]
    selected_q = 'MIN' if selected_q == 'MAX' else selected_q # to avoid submitting MAX results.
    return selected_q


def load_facet_to_topic_dict(topic_file_path):
    topic_df = pd.read_csv(topic_file_path, sep='\t')
    facet_to_topic_dict = topic_df.set_index('facet_id')['topic_id'].to_dict()
    return facet_to_topic_dict


def load_eval_dict(eval_file_path, topic_file_path):
    topic_df = pd.read_csv(topic_file_path, sep='\t')
    facet_array = topic_df['facet_id'].values
    with open(eval_file_path, 'rb') as fi:
        eval_dict = pickle.load(fi)
    # we keep only the instances in the topic file.
    new_eval_dict = {}
    for metric in eval_dict:
        new_eval_dict[metric] = {}
        for fid in eval_dict[metric]:
            if fid in facet_array:
                new_eval_dict[metric][fid] = eval_dict[metric][fid]
    return new_eval_dict


def load_run_dict_doc_relevance(run_file):
    run_df = pd.read_csv(run_file, sep=' ', header=None)
    run_df = run_df.sort_values(by=4).drop_duplicates(subset=[0], keep='last')  # we only keep the top ranked question.
    run_dict = run_df.set_index(0)[2].to_dict()  # we convert the run dataframe to dict.
    return run_dict


def _cmp_func(x, y):
    if x['score'] < y['score']:
        return -1
    if x['score'] == y['score']:
        return 0
    else:
        return 1

def get_best_q(performance_dict_list):
    sorted_performance = sorted(performance_dict_list, key=functools.cmp_to_key(_cmp_func), reverse=True)
    best_q = []
    best_score = sorted_performance[0]['score']
    for q in performance_dict_list:
        if q['score'] == best_score:
            best_q.append(q['qid'])
    return best_q

def replace(sentence, lower_case=True):
    if str(sentence) == "nan":
        return "<None>"
    if sentence.endswith('.'):
        sentence = sentence.split('.')[0] + ' .'
    elif sentence.endswith('?'):
        sentence = sentence.split('?')[0] + ' ?'
    if lower_case:
        sentence = sentence.lower()
    # Replace words like gooood to good
    sentence = re.sub(r'(\w)\1{2,}', r'\1\1', sentence)
    # Normalize common abbreviations
    words = sentence.split(' ')
    words = [replacement[word] if word in replacement else word for word in words]
    sentence_repl = " ".join(words)
    return sentence_repl

def process_train_eval(data_dir, experiment_type="dev"):
    eval_file_path = "%s/single_turn_train_eval.pkl" % data_dir
    topic_file_path = "%s/%s.tsv" % (data_dir, experiment_type)

    eval_dict = load_eval_dict(eval_file_path, topic_file_path)
    facet_to_topic_dict = load_facet_to_topic_dict(topic_file_path)
    data = pd.read_csv(topic_file_path, sep='\t')
    
    best_topic_q = {}
    for tid in data['topic_id'].unique():
        question_ids = data.loc[data['topic_id']==tid, 'question_id'].unique().tolist()
        q_score = []
        for qid in question_ids:
            run_dict = {}
            run_dict[tid] = qid
            performance = evaluate_document_relevance(eval_dict, facet_to_topic_dict, run_dict)
            run_score = 0.0
            for k in performance:
                run_score += performance[k]
                val = {
                    'qid': qid, 
                    'score': run_score
                }
            q_score.append(val)
        best_q = get_best_q(q_score)
        best_topic_q[tid] = best_q[0]
    for k, v in best_topic_q.items():
        print("topic_id: {} best_q: {}".format(k, v))
    
    outputs = []
    for tid in data['topic_id'].unique():
        request = str(data.loc[data['topic_id']==tid, 'initial_request'].tolist()[0])
        request = replace(request, lower_case=True)
        clari_need = str(data.loc[data['topic_id']==tid, 'clarification_need'].tolist()[0])
        question_ids = data.loc[data['topic_id']==tid, 'question_id'].unique().tolist()
        questions = data.loc[data['topic_id']==tid, 'question'].unique().tolist()
        assert len(question_ids) == len(questions)
        pos_qid = best_topic_q[tid]
        for i, qid in enumerate(question_ids):
            if qid == pos_qid:
                label = 1
            else:
                label = 0
            question = replace(questions[i], lower_case=True)
            sample = {
                "topic_id": tid,
                "initial_request": request,
                "clarification_need": clari_need,
                "question_id": qid,
                "question": question,
                "rank_label": label
            }
            outputs.append(sample)
    save_file_path = "%s/%s.data.txt" % (data_dir, experiment_type)
    with open(save_file_path, "w") as fo:
        fo.write("topic_id\tinitial_request\tclarification_need\tquestion_id\tquestion\trank_label\n")
        for sample in outputs:
            fo.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sample["topic_id"], sample["initial_request"], 
                     sample["clarification_need"], sample["question_id"], sample["question"], sample["rank_label"]))
    print("saved data to %s" % save_file_path)

def _validate(query):
    valid_query = str(query).strip()
    remove_str = ['*', '?', '!', ':', '-', '(', ')', '[', ']', '{', '}']
    for s in remove_str:
        if s in valid_query:
            valid_query = valid_query.replace(s, '')

    return valid_query


def process_test(data_dir, index_dir, top_k=30):
    queryer = Queryer(index_dir=index_dir, top_k=top_k)
    with open("%s/id2question.json" % index_dir, "r") as fr:
        id2question = json.load(fr)
    test_file_path = "%s/test.tsv" % data_dir
    data = pd.read_csv(test_file_path, sep='\t')

    outputs = []
    for tid in data['topic_id'].unique():
        request = str(data.loc[data['topic_id']==tid, 'initial_request'].tolist()[0])
        request = replace(request, lower_case=True)
        query = _validate(request)
        result = queryer.run_query(query)
        result_ids = result['ids']
        candidate_id_k = []
        candidate_k = []
        for idx in result_ids:
            sent = id2question[idx]
            candidate_id_k.append(idx)
            candidate_k.append(sent)
        assert len(candidate_id_k) == len(candidate_k)
        for qid, q in zip(candidate_id_k, candidate_k):
            sample = {
                "topic_id": tid,
                "initial_request": request,
                "question_id": qid,
                "question": q
            }
            outputs.append(sample)
    
    save_file_path = "%s/test.data.txt" % (data_dir)
    with open(save_file_path, "w") as fo:
        fo.write("topic_id\tinitial_request\tquestion_id\tquestion\n")
        for sample in outputs:
            fo.write("{}\t{}\t{}\t{}\n".format(sample["topic_id"], sample["initial_request"], 
                    sample["question_id"], sample["question"]))
    print("saved data to %s" % save_file_path)



if __name__ == "__main__":
    data_dir = "./data"
    index_dir = "./index"
    
    # process train/dev data
    for exp in ["train", "dev"]:
        process_train_eval(data_dir=data_dir, experiment_type=exp)
    
    # process test data
    process_test(data_dir=data_dir, index_dir=index_dir, top_k=30)
