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

parser = argparse.ArgumentParser()
parser.add_argument("--q_p_file", default=None, help="Question Paragraph pickle file")
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
parser.add_argument("--output_dir", default=None, help="Directory to write the trained weights")
parser.add_argument("--num_paragraphs_per_question", default=4, help="Number of top k paragraphs should match the one chosen in prepare_dataset.py")
args = parser.parse_args()

logger = logging.getLogger(__name__)

## read the pickle file
q_p = pickle.load(open(args.q_p_file, 'rb'))

### create BERT input examples  template
train_InputExamples = []

for q in q_p:
    for p in q.paragraphs:
        train_InputExamples.append(bert_utils.InputExample(p.tokens, q.tokens, p.start_index, p.end_index))

print ()
print ("Length of train_InputExamples: ", len(train_InputExamples))
print ()

####################

### Some defaults:

bert_model = 'bert-base-uncased'
# bert_model = 'bert-large-uncased-whole-word-masking'

## use this when you want to begin with Squad initilaization
##pretrained_dir = args.output_dir + "/small/checkpoint-22000"

MAX_SEQ_LENGTH = 384
do_lower_case = True
batch_size = 48  # 64
learning_rate = 3e-5
adam_epsilon = 1e-8
num_epochs = 3
warmup_proportion = 0.0
gradient_accumulation_steps = 1  # 4
seed = 5
logging_steps = 50
save_steps = 4000
max_grad_norm = 1.0

num_paragraphs_per_question = int(args.num_paragraphs_per_question)
####################

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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), False))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

num_train_optimization_steps = int(len(train_InputExamples) / batch_size / gradient_accumulation_steps) * num_epochs

tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
model_qa = BertQA.from_pretrained(bert_model, cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                     'distributed_{}'.format(args.local_rank)))

## Use this when you want to begin with Squad model initialization:
##tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case = do_lower_case)
##model_qa = BertQA.from_pretrained(pretrained_dir)


if args.local_rank == 0:
    torch.distributed.barrier()

model_qa.to(device)

train_features = bert_utils.convert_examples_to_features(train_InputExamples, MAX_SEQ_LENGTH, tokenizer)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_start_positions = torch.tensor([f.start_label_ids for f in train_features], dtype=torch.long)
all_end_positions = torch.tensor([f.end_label_ids for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)

train_sampler = SequentialSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()

param_optimizer = list(model_qa.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
warmup_steps = int(num_train_optimization_steps * warmup_proportion)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

if n_gpu > 1:
    model_qa = torch.nn.DataParallel(model_qa)

if args.local_rank != -1:
    model_qa = torch.nn.parallel.DistributedDataParallel(model_qa, device_ids=[args.local_rank],
                                                         output_device=args.local_rank,
                                                         find_unused_parameters=True)
global_step = 0
tr_loss, logging_loss = 0.0, 0.0
model_qa.zero_grad()
total_loss = []
train_iterator = trange(int(num_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

logsoftmax = torch.nn.LogSoftmax(dim=-1)

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

        model_qa.train()
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        start_logits, end_logits = model_qa(input_ids, input_mask, segment_ids)

        ### compute shared normalized loss

        start_logit_chunks = torch.split(start_logits, num_paragraphs_per_question, dim=0)
        end_logit_chunks = torch.split(end_logits, num_paragraphs_per_question, dim=0)
        start_position_chunks = torch.split(torch.nn.functional.one_hot(start_positions, num_classes=MAX_SEQ_LENGTH),
                                            num_paragraphs_per_question, dim=0)
        end_position_chunks = torch.split(torch.nn.functional.one_hot(end_positions, num_classes=MAX_SEQ_LENGTH),
                                          num_paragraphs_per_question, dim=0)
        num_chunks = len(start_logit_chunks)
        loss_chunks = []
        for n in range(num_chunks):
            start_logit_list = []
            end_logit_list = []
            start_position_list = []
            end_position_list = []
            num_no_answer_start = 0
            num_no_answer_end = 0
            for p in range(num_paragraphs_per_question):
                start_logit_list.append(start_logit_chunks[n][p])
                end_logit_list.append(end_logit_chunks[n][p])

                ### add a line so that only gold paragraph has value 1 in one_hot for start and end
                ### that is every question has answer, zeroth index should never be 1, for squad v1.1
                ### Add some more lines so that if question has no answer, mark the zeroth index of the 1st paragraph
                ### This will make it compatible with Squad v2.0 and NewsQA

                temp_start_pos = start_position_chunks[n][p]
                temp_end_pos = end_position_chunks[n][p]
                if temp_start_pos[0].item() == 1:
                    temp_start_pos[0] = 0
                    num_no_answer_start += 1
                if temp_end_pos[0].item() == 1:
                    temp_end_pos[0] = 0
                    num_no_answer_end += 1
                start_position_list.append(temp_start_pos)
                end_position_list.append(temp_end_pos)

            ### mark the first paragraph's CLS token as prediction for noAnswer, this time the zeroth index is 1!
            if num_no_answer_start == num_paragraphs_per_question and num_no_answer_end == num_paragraphs_per_question:
                start_position_list[0][0] = 1
                end_position_list[0][0] = 1

            f_start_logit_tensor = torch.cat(start_logit_list).cpu()
            f_end_logit_tensor = torch.cat(end_logit_list).cpu()
            f_start_position_tensor = torch.cat(start_position_list).cpu()
            f_start_position_tensor = f_start_position_tensor.type(torch.FloatTensor)
            f_end_position_tensor = torch.cat(end_position_list).cpu()
            f_end_position_tensor = f_end_position_tensor.type(torch.FloatTensor)
            start_loss = -torch.sum(f_start_position_tensor * logsoftmax(f_start_logit_tensor))
            end_loss = -torch.sum(f_end_position_tensor * logsoftmax(f_end_logit_tensor))
            loss_chunks.append((start_loss + end_loss) / 2.0)
        loss = sum(loss_chunks) / len(loss_chunks)

        #### shared normalized loss computation ends for one batch

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_qa.parameters(), max_grad_norm)
        total_loss.append(loss.item())
        tr_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model_qa.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                print("****** Loss  ******** : ", (tr_loss - logging_loss) / logging_steps)
                print ("****** Average Loss ********: ", sum(total_loss) / len(total_loss))
                logging_loss = tr_loss

            if args.local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model_qa.module if hasattr(model_qa, 'module') else model_qa
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

tb_writer.close()
logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)
print ()
print ("************** Training Complete **************")
print ()
print ()
output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logger.info("Saving final model checkpoint to %s", output_dir)
model_to_save = model_qa.module if hasattr(model_qa, 'module') else model_qa
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

# pickle.dump(total_loss, open('loss_list.pkl', 'wb'))
