import csv
import os
import torch
import numpy as np
import logging
from torch.utils.data import TensorDataset
from transformers import BertTokenizer


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, text_pattern=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_pattern = text_pattern
        self.label = label

class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self, input_ids, input_mask, segment_ids, input_pattern, label_id, input_len):
        self.input_ids   = input_ids
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id
        self.input_pattern = input_pattern
        self.input_len = input_len


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,vocab_path,do_lower_case):
        self.tokenizer = BertTokenizer(vocab_path,do_lower_case)

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self, lines, example_type):
        '''
        lines: a list of dtype tuple
        '''
        examples = []
        for i,line in enumerate(lines):
            guid = '%s-%d'%(example_type,i)
            if isinstance(line[0], tuple):
                text_a = line[0][0]
                text_pattern = line[0][1]
                text_b = None
            elif isinstance(line[0], str):
                text_a = line[0]
                text_pattern = None
                text_b = None
            else:
                raise ValueError("data format error for creating examples")
            label = line[1]
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, \
                text_pattern=text_pattern, label=label)
            examples.append(example)
        return examples

    def create_features(self, examples, max_seq_len):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''
        features = []
        for ex_id,example in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)
            tokens_b = None
            text_pattern = example.text_pattern
            label_id = example.label

            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self.truncate_seq_pair(tokens_a,tokens_b,max_length = max_seq_len - 3)
            else:
                # Account for [CLS] and [SEP] with '-2'
                if len(tokens_a) > max_seq_len - 2:
                    tokens_a = tokens_a[:max_seq_len - 2]
            tokens = ['[CLS]'] + tokens_a + ['[SEP]']
            segment_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b + ['[SEP]']
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_len - len(input_ids))
            input_len = len(input_ids)

            input_ids   += padding
            input_mask  += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            if ex_id < 4:
                logging.info("*** Example ***")
                logging.info(f"guid: {example.guid}" % ())
                logging.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                logging.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                logging.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                logging.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
                if text_pattern:
                    logging.info(f"input_pattern: {' '.join([str(x) for x in text_pattern])}")


            feature = InputFeature(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    input_pattern=text_pattern,
                                    label_id=label_id,
                                    input_len=input_len)
            features.append(feature)
        return features

    def create_dataset(self, features, with_pattern=False, is_sorted=False):
        # Convert to Tensors and build dataset
        if is_sorted:
            logging.info("sorted data by th length of input")
            features = sorted(features, key=lambda x:x.input_len, reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features],dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        if with_pattern:
            all_input_pattern = torch.tensor([f.input_pattern for f in features], dtype=torch.float)
        else:
            all_input_pattern = torch.tensor([0], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, \
                all_input_pattern, all_label_ids, all_input_lens)
        return dataset