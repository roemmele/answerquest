import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss


class BertQA(BertPreTrainedModel):

    def __init__(self, config):
        super(BertQA, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


def loss(start_logits, end_logits, start_positions, end_positions):
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)

    loss_fn = CrossEntropyLoss()
    start_loss = loss_fn(start_logits, start_positions)
    end_loss = loss_fn(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2

    return total_loss
