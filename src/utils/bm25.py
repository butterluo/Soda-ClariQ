from rank_bm25 import BM25Okapi
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer


def stem_tokenize(text, remove_stopwords=True):
    stemmer = PorterStemmer()
    tokens = [word for sent in nltk.sent_tokenize(text) \
                                      for word in nltk.word_tokenize(sent)]
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return [stemmer.stem(word) for word in tokens]


def get_top_n(query, question_bank, top_n=5):
    question_bank = question_bank.fillna('')
    question_bank['tokenized_question_list'] = question_bank['question'].map(stem_tokenize)
    question_bank['tokenized_question_str'] = question_bank['tokenized_question_list'].map(lambda x: ' '.join(x))
    
    bm25_corpus = question_bank['tokenized_question_list'].tolist()
    bm25 = BM25Okapi(bm25_corpus)
    bm25_ranked_list = bm25.get_top_n(stem_tokenize(query, True), 
                                    bm25_corpus, 
                                    n=top_n)
    bm25_q_list = [' '.join(sent) for sent in bm25_ranked_list]
    preds = question_bank.set_index('tokenized_question_str').loc[bm25_q_list, 'question_id'].tolist()
    return preds
    