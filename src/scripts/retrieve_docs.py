import os
import pickle
import uuid
from urllib import request
from bs4 import BeautifulSoup
import nltk
import pandas as pd


def retrieve(doc_id):
    if str(doc_id).startswith("clueweb09"):
        prefix = "clueweb09"
        index = "cw09"
    elif str(doc_id).startswith("clueweb12"):
        prefix = "clueweb12"
        index = "cw12"
    else:
        raise ValueError("Doc index unknown!")
    name = "%s:%s" % (prefix, doc_id)
    doc_uuid = uuid.uuid5(uuid.NAMESPACE_URL, name)
    url = "https://www.chatnoir.eu/cache?uuid=%s&index=%s&raw&plain" % (doc_uuid, index)
    print("url:", url)
    try:
        html = request.urlopen(url).read()
        soup = BeautifulSoup(html, 'html.parser')
        content = str(soup.find('body'))
    except:
        content = ""
    return content


def retrieve_for_data(data_dir, output_dir, data_type="train", topk=20):
    top10k_docs_path = "%s/top10k_docs_dict.pkl" % data_dir
    with open(top10k_docs_path, "rb") as fr:
        top10k_docs = pickle.load(fr)
    data_path = "%s/%s.tsv" % (data_dir, data_type)
    data = pd.read_csv(data_path, sep='\t')
    
    for tid in data['topic_id'].unique():
        print("tid:", tid)
        top_dids = top10k_docs[tid][:topk]
        for idx, did in enumerate(top_dids):
            doc = retrieve(doc_id=did)
            output_path = "%s/topic-%s-top%d.doc.txt" % (output_dir, tid, idx+1)
            with open(output_path, 'w', encoding='utf-8') as fo:
                fo.write(doc)
            print("saved data to [%s]" % output_path)
        break


def main(data_dir, output_dir):
    '''
    train_qrel = "%s/train.qrel" % data_dir
    facet_docs = {}
    with open(train_qrel, "r") as fr:
        for idx, line in enumerate(fr):
            if idx == 0:
                break
            facet_id, _, doc_id, score = line.strip().split()
            if int(score) >= 1:
                if not facet_id in facet_docs:
                    facet_docs[facet_id] = [doc_id]
                else:
                    facet_docs[facet_id].append(doc_id)
    print(facet_docs)
    for fid, doc_ids in facet_docs.items():
        print("facet_id:", fid)
        for did in doc_ids:
            doc = retrieve(doc_id=did)
            print("doc:", did)
            print(doc)
    '''
    retrieve_for_data(data_dir, output_dir, data_type="train", topk=100)
    #retrieve_for_data(data_dir, data_type="dev")
    #retrieve_for_data(data_dir, data_type="test")


if __name__ == "__main__":
    data_dir = "./data"
    output_dir = "./docs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(data_dir, output_dir)
