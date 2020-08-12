import json
import argparse
import sys
import logging
import random
from datetime import datetime
import os
import numpy as np
import pickle

from tqdm import tqdm, trange
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

import bert_utils
import model
from model import BertQA

from transformers import BertTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, WarmupLinearSchedule

from tensorboardX import SummaryWriter
from prepare_dataset import QuestionParagraph, ParagraphAnswer


def return_original_index(index, max_seq_length, num_paragraphs):
    for i in range(0, num_paragraphs):
        if index >= max_seq_length * i and index < max_seq_length * (i + 1):
            return (i, index - (max_seq_length * i))


parser = argparse.ArgumentParser()
parser.add_argument("--q_p_file", default=None, help="Question Paragraph pickle file")
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
parser.add_argument("--output_dir", default=None, help="Directory to write the trained weights")
parser.add_argument("--num_paragraphs_per_question", default=None, help="Num paragraphs per question should match the one chosen in prepare_dataset.py")
args = parser.parse_args()

## read the pickle file
q_p = pickle.load(open(args.q_p_file, 'rb'))

### create BERT input examples  template
dev_InputExamples = []

for q in q_p:
    for p in q.paragraphs:
        dev_InputExamples.append(bert_utils.InputExample(p.tokens, q.tokens, p.start_index, p.end_index))

print ()
print ("Length of dev_InputExamples: ", len(dev_InputExamples))
print ()

####################

### Some defaults:

# bert_model = 'bert-base-uncased'
# bert_model = 'bert-large-uncased-whole-word-masking'

MAX_SEQ_LENGTH = 384
do_lower_case = True
batch_size = 180 * 4  # 192 * 4
learning_rate = 3e-5
adam_epsilon = 1e-8
num_epochs = 5
warmup_proportion = 0.0
gradient_accumulation_steps = 1
seed = 5
logging_steps = 50
save_steps = 1000
max_grad_norm = 1.0

num_paragraphs_per_question = int(args.num_paragraphs_per_question)
#####################################


# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # print ('local rank: ', args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = 1

print ()
print ("Number of GPUs: ", n_gpu)
print ()

batch_size = batch_size // gradient_accumulation_steps

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=do_lower_case)
model_qa = BertQA.from_pretrained(args.output_dir)
model_qa.to(device)

dev_features = bert_utils.convert_examples_to_features(dev_InputExamples, MAX_SEQ_LENGTH, tokenizer)
all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
all_start_positions = torch.tensor([f.start_label_ids for f in dev_features], dtype=torch.long)
all_end_positions = torch.tensor([f.end_label_ids for f in dev_features], dtype=torch.long)
dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)

dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(train_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

if n_gpu > 1:
    model_qa = torch.nn.DataParallel(model_qa)

log_softmax = torch.nn.LogSoftmax()
ctr = 0

start_list = []
end_list = []

for batch in tqdm(dev_dataloader, desc="Evaluating"):
    model_qa.eval()
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        start_logits, end_logits = model_qa(input_ids, input_mask, segment_ids)

    ### split into parts
    start_logits = start_logits.detach().cpu()
    end_logits = end_logits.detach().cpu()

    start_logit_chunks = torch.split(start_logits, num_paragraphs_per_question, dim=0)
    end_logit_chunks = torch.split(end_logits, num_paragraphs_per_question, dim=0)
    num_chunks = len(start_logit_chunks)
    for n in range(num_chunks):
        start_logit_list = []
        end_logit_list = []
        for p in range(num_paragraphs_per_question):
            start_logit_list.append(start_logit_chunks[n][p])
            end_logit_list.append(end_logit_chunks[n][p])
        # print ('start logit list shape: ', torch.cat(start_logit_list).size())
        f_start_index = torch.argmax(log_softmax(torch.cat(start_logit_list)))
        f_end_index = torch.argmax(log_softmax(torch.cat(end_logit_list)))
        s_p, s_index = return_original_index(f_start_index, MAX_SEQ_LENGTH, num_paragraphs_per_question)
        e_p, e_index = return_original_index(f_end_index, MAX_SEQ_LENGTH, num_paragraphs_per_question)
        start_list.append((s_p, s_index))
        end_list.append((e_p, e_index))
    ctr += 1
    print ("Done batch: ", ctr)

actual_ans = []
pred_ans = []
not_in_same_paragraph = 0

for index, q in enumerate(q_p):
    actual_ans.append(q.original_consensus_answer)
    q_tokens = q.tokens
    s_p, s_index = start_list[index]
    e_p, e_index = end_list[index]
    if s_p != e_p:
        not_in_same_paragraph += 1
        pred_ans.append("")
        continue
    new_s_index = s_index - len(q_tokens) - 2
    new_e_index = e_index - len(q_tokens) - 2
    orig_p = q.paragraphs[s_p].whitespace_tokens
    p = q.paragraphs[s_p].tokens

    if new_s_index < 0 or new_s_index > len(p) - 1 or new_e_index < 0 or new_e_index > len(p) - 1:
        pred_ans.append("")
    else:
        if q.paragraphs[s_p].tokens_to_original_index == None:
            pred_ans.append("")
        else:
            orig_s_index = q.paragraphs[s_p].tokens_to_original_index[new_s_index]
            orig_e_index = q.paragraphs[s_p].tokens_to_original_index[new_e_index]
            pred_ans.append(" ".join(orig_p[orig_s_index:orig_e_index + 1]))

import pickle

print ()
print ("Not in same paragraph: ", not_in_same_paragraph)
print ()
pickle.dump(actual_ans, open('actual_ans.pkl', 'wb'))
pickle.dump(pred_ans, open('pred_ans.pkl', 'wb'))
