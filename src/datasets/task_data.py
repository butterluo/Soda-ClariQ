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
    
    def extract_pattern(self, sentence):
        tokens = sentence.split()
        # feature of length
        if len(tokens) < 3:
            length_f = 1
        elif len(tokens) < 8:
            length_f = 2
        else:
            length_f = 3
        # feature of special words
        # [what, how, when, where, who, tell...about..., find/give/need...information...,looking for...]
        sent_f = [0, 0, 0, 0, 0, 0, 0, 0]
        if "what" in sentence:
            sent_f[0] = 1
        if "how" in sentence:
            sent_f[1] = 1
        if "when" in sentence:
            sent_f[2] = 1
        if "where" in sentence:
            sent_f[3] = 1
        if "who" in sentence:
            sent_f[4] = 1
        if "tell" in sentence and "about" in sentence:
            sent_f[5] = 1
        if "information" in sentence:
            if "find" in sentence or "give" in sentence or "need" in sentence:
                sent_f[6] = 1
        if "looking" in sentence and "for" in sentence:
            sent_f[7] = 1
        pattern_f = [length_f] + sent_f
        return pattern_f


    def read_data(self, data_path, is_train=False, with_pattern=False, shuffle=False):
        data = pd.read_csv(data_path, sep='\t')
        sentences, targets = [], []
        data_list = []
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
            
            if with_pattern:
                pattern_feature = self.extract_pattern(request)
                sentences.append((request, pattern_feature))
            else:
                sentences.append(request)
            targets.append(target)
        
        for data_x, data_y in zip(sentences, targets):
            data_list.append((data_x, data_y))
        if shuffle:
            random.shuffle(data_list)
        return data_list


    def save_predict(self, raw_data_path, result_list, save_dir):
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
                fw.write("{} {}\n".format(tid, cln))
        print("save result to [%s]" % save_file)
        