import random
import pandas as pd


class TaskData(object):
    def __init__(self, task):
        self.task = task
        assert self.task in ["task1", "task2"]
    
    def get_task(self):
        return self.task
    
    def get_num_labels(self):
        if self.task == "task1":
            num_labels = 4
        else:
            num_labels = 2
        return num_labels
    
    def read_data(self, data_path, is_train=False, shuffle=False):
        data = pd.read_csv(data_path, sep='\t')
        sentences, targets = [], []
        data_list = []
        if self.task == "task1":
            for tid in data['topic_id'].unique():
                request = str(data.loc[data['topic_id']==tid, 'initial_request'].tolist()[0])
                if is_train:
                    clari_need = str(data.loc[data['topic_id']==tid, 'clarification_need'].tolist()[0])
                    assert clari_need in ['1', '2', '3', '4']
                    if clari_need == '1':
                        target = 0
                    elif clari_need == '2':
                        target = 1
                    elif clari_need == '3':
                        target = 2
                    else:
                        target = 3                    
                else:
                    target = -1
                sentences.append(request)
                targets.append(target)
        else:
            for tid in data['topic_id'].unique():
                request = str(data.loc[data['topic_id']==tid, 'initial_request'].tolist()[0])
                questions = data.loc[data['topic_id']==tid, 'question'].tolist()
                if is_train:
                    labels = data.loc[data['topic_id']==tid, 'rank_label'].tolist()
                    assert len(labels) == len(questions)
                else:
                    labels = [-1 for _ in range(len(questions))]
                for q, label in zip(questions, labels):
                    sentences.append((request, q))
                    
                    targets.append(label)
        for data_x, data_y in zip(sentences, targets):
            data_list.append((data_x, data_y))
        if shuffle:
            random.shuffle(data_list)
        return data_list


    def save_predict(self, raw_data_path, result_list, save_dir):
        if self.task == "task1":
            data = pd.read_csv(raw_data_path, sep='\t')
            topic_ids = data['topic_id'].unique()
            clari_needs = [r+1 for r in result_list]
            print("len topic_id:", len(topic_ids))
            print("len clari_needs:", len(clari_needs))
            assert len(topic_ids) == len(clari_needs)

            if "dev" in raw_data_path:
                save_file = "%s/dev_clari_need.txt" % save_dir
            else:
                save_file = "%s/test_clari_need.txt" % save_dir
            with open(save_file, 'w') as fw:
                for tid, cln in zip(topic_ids, clari_needs):
                    print("{} {}".format(tid, cln))
                    fw.write("{} {}\n".format(tid, cln))
            print("save result to [%s]" % save_file)
        else:
            data = pd.read_csv(raw_data_path, sep='\t')
            requests = data['initial_request'].tolist()
            questions = data['question'].tolist()
            print("requests:", len(requests))
            print("questions:", len(questions))
            print("result_list:", len(result_list))

            for idx, req in enumerate(requests):
                print("{}\t{}\t{}".format(req, questions[idx], result_list[idx]))
            