import warnings
warnings.filterwarnings('ignore')

import numpy
import os
import string
import re
import spacy
import torch
from collections import Counter
from question_answering import document_bert_inference as br

torch.manual_seed(0)

model = None
tokenizer = None
device = None


def init_qa_pipeline(model_path):
    global model
    global tokenizer
    global device
    model, tokenizer, device = br.load_model(model_path)


def answer_questions(texts, questions, max_seq_length=384, batch_size=256):
    answers = br.batch_inference(questions, texts, model, tokenizer, device, max_seq_length, batch_size)
    return answers


def strip_answer(answer):
    '''Remove whitespace and leading/trailing punctuation'''
    answer = answer.strip()
    answer = re.sub('[{}]+$'.format(re.escape('.,!?:;') + '\s'), '',  answer)
    answer = re.sub('^[{}]+'.format(re.escape('.,!?:;') + '\s'), '',  answer)
    return answer


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_precision_recall_f1(pred_answer, correct_answer):
    prediction_tokens = normalize_answer(pred_answer).split()
    ground_truth_tokens = normalize_answer(correct_answer).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate(pred_answers, all_correct_answers):

    precision_scores = []
    recall_scores = []
    f1_scores = []
    for pred_answer, correct_answers in zip(pred_answers, all_correct_answers):
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        # If multiple correct answers given, compared predicted answer to all and take best score
        for correct_answer in correct_answers:
            precision, recall, f1 = get_precision_recall_f1(pred_answer, correct_answer)
            if f1 >= best_f1:
                best_precision = precision
                best_recall = recall
                best_f1 = f1
        precision_scores.append(best_precision)
        recall_scores.append(best_recall)
        f1_scores.append(best_f1)
    mean_precision = numpy.mean(precision_scores)
    mean_recall = numpy.mean(recall_scores)
    mean_f1 = numpy.mean(f1_scores)

    return {'precision': mean_precision, 'recall': mean_recall, 'f1': mean_f1}
