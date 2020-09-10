import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertForClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Here 9 denotes the pattern feature length
        self.clf_layer = nn.Linear(config.hidden_size + 9, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, 
                attention_mask=None, head_mask=None, input_pattern=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        final_output = torch.cat((pooled_output, input_pattern), 1)
        logits = self.clf_layer(final_output)
        return logits
