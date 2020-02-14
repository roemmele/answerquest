import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(root_dir, "question_answering/demo_inference/"))

import argparse
import json
import numpy
import warnings
warnings.filterwarnings('ignore')

import spacy
import torch
from onmt.bin.translate import translate, _get_parser
from onmt.translate.translator import build_translator
from transformers import GPT2Tokenizer
from flask import Flask, request, render_template, jsonify
from question_generation import utils
from question_answering.demo_inference import document_bert_inference


app = Flask(__name__)

spacy_model = spacy.load('en')

torch.manual_seed(0)

# QG and QA pipeline components are initialized in main function below
qg_tokenizer = None
qg_opt = None
qg_model = None

qa_tokenizer = None
qa_model = None
qa_device = None


def init_qg_pipeline(tokenizer_path, model_path, batch_size=256, gpu_id=-1, beam_size=5):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    parser = _get_parser()
    opt = parser.parse_args(args=['-model', model_path,
                                  '-src', "",
                                  '-batch_size', str(batch_size),
                                  '-beam_size', str(beam_size),
                                  '-gpu', str(gpu_id),
                                  ])
    model = build_translator(opt, report_score=True)
    print("Loaded all model components\n")
    return tokenizer, opt, model


def init_qa_pipeline(model_path):
    model, tokenizer, device = document_bert_inference.load_model(model_path)
    return model, tokenizer, device


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/qg_with_qa', methods=['POST'])
def generate_and_answer_questions():
    try:
        #import pdb;pdb.set_trace()
        input_text = request.values['input_text']
        spacy_input_sents = list(spacy_model(input_text).sents)
        qg_input_sents = [sent.text for sent in spacy_input_sents]

        sent_idxs, questions, scores = annotate_generate_filter_questions(spacy_input_sents)

        # For QA, look for answer to generated question within a 3-sentence context chunk.
        # This context consists of the sentence that generated the question and the sentences immediately before and after this sentence.
        qa_input_chunks = [" ".join(qg_input_sents[max(0, sent_idx - 1):sent_idx + 2])
                           for sent_idx in sent_idxs]
        answers = answer_questions(qa_input_chunks, questions)
        # Filter questions that don't have answers
        items = [(sent_idx, question, answer) for sent_idx, question, answer
                 in zip(sent_idxs, questions, answers) if answer]
        if not items:  # No answerable questions found
            questions = ["No Q&A pairs found. Try using a longer text."]
            answers = [""]
        else:
            sent_idxs, questions, answers = zip(*items)
            sent_idxs, questions, answers = filter_questions_with_duplicate_answers(sent_idxs, questions, answers)
            sent_idxs, questions, answers = filter_questions_by_sent_idx(sent_idxs, questions, answers)

    except Exception as e:
        print(e)
        questions = ["An error occurred. Try using a shorter text."]
        answers = [""]
    # Release memory
    torch.cuda.empty_cache()
    return jsonify({'questions': questions, 'answers': answers})


def filter_questions_by_sent_idx(sent_idxs, questions, answers):
    '''Only return one question-answer item per original input sentence'''

    # Regroup questions by sentence index and select first question in each group that has answer
    questions_by_sent_idx = {}
    answers_by_sent_idx = {}
    for sent_idx, question, answer in zip(sent_idxs, questions, answers):
        if answer and sent_idx not in answers_by_sent_idx:
            questions_by_sent_idx[sent_idx] = question
            answers_by_sent_idx[sent_idx] = answer
    sent_idxs, questions = zip(*questions_by_sent_idx.items())
    answers = list(answers_by_sent_idx.values())
    return sent_idxs, questions, answers


def filter_questions_with_duplicate_answers(sent_idxs, questions, answers):
    '''Filter questions that have the same answer - keep the queston that appears first in the list'''
    answers_to_questions = {}
    for sent_idx, question, answer in zip(sent_idxs, questions, answers):
        if answer not in answers_to_questions:
            answers_to_questions[answer] = (sent_idx, question)

    sent_idxs, questions, answers = zip(*[(sent_idx, question, answer) for answer, (sent_idx, question)
                                          in answers_to_questions.items()])
    return sent_idxs, questions, answers


def filter_duplicate_questions(sent_idxs, questions, scores):  # , other_segments=[]):
    '''Keep the duplicate with the highest score.'''
    questions_to_scores = {}
    # For each question, save max score
    for q_idx, (sent_idx, question, score) in enumerate(zip(sent_idxs, questions, scores)):
        if (question not in questions_to_scores
                or score > questions_to_scores[question][-1]):
            questions_to_scores[question] = (q_idx, sent_idx, score)
    unique_q_idxs = sorted([q_idx for question, (q_idx, sent_idx, score) in questions_to_scores.items()])
    sent_idxs, questions, scores = zip(*[(sent_idxs[q_idx], questions[q_idx], scores[q_idx])
                                         for q_idx in unique_q_idxs])
    return sent_idxs, questions, scores  # , other_segments


def get_answer_annotated_input_sents(spacy_input_sents):
    grouped_input_sents = []  # A group is all annotated sents that map to the same original input sentence
    grouped_answers = []
    for sent in spacy_input_sents:
        input_sents, answers = utils.extract_candidate_answers(sent)
        grouped_input_sents.append(input_sents)
        grouped_answers.append(answers)
    return grouped_input_sents, grouped_answers


def annotate_generate_filter_questions(spacy_input_sents, top_k=3):
    '''Mark all candidate answer spans (currently uses noun chunks and named entities as answer spans)
    in the input sents, which will result in multiple annotated sentences per original input sentence, each having a unqiue
    answer span. Provide all resulting sentences to the QG model.
    For each group of annotated sentences derived from the same original input sentence, 
    only keep the generated question with the highest score'''

    # Get annotated inputs
    grouped_input_sents, _ = get_answer_annotated_input_sents(spacy_input_sents)
    # Add unannotated inputs
    grouped_input_sents = [annotated_sents + [spacy_sent.text] for annotated_sents, spacy_sent
                           in zip(grouped_input_sents, spacy_input_sents)]
    # Flatten sentences so QG can do batching
    n_sents_per_group = [len(sents) for sents in grouped_input_sents]
    group_boundary_idxs = [0]
    for n_sents in n_sents_per_group:
        group_boundary_idxs.append(group_boundary_idxs[-1] + n_sents)
    input_sents = [sent for sents in grouped_input_sents
                   for sent in sents]
    questions, scores = generate_questions(input_sents)
    # Reshape questions/scores into groups
    grouped_questions, grouped_scores = zip(*[(questions[start_idx:end_idx], scores[start_idx:end_idx])
                                              for start_idx, end_idx in zip(group_boundary_idxs[:-1],
                                                                            group_boundary_idxs[1:])])
    # For each group of questions derived from same input, select top_k questions by score
    grouped_score_idxs = [numpy.argsort(score_group)[::-1][:top_k]  # if score_group else -1
                          for score_group in grouped_scores]
    grouped_questions, grouped_scores = zip(*[([question_group[idx] for idx in idxs_group],
                                               [score_group[idx] for idx in idxs_group])  # if max_score_idx > -1 else ("", None)
                                              for question_group, score_group, idxs_group in
                                              zip(grouped_questions, grouped_scores, grouped_score_idxs)])
    # Flatten questions/scores and make separate list for their corresponding sentence indices
    sent_idxs, questions, scores = zip(*[(sent_idx, question, score) for sent_idx, (question_group, score_group)
                                         in enumerate(zip(grouped_questions, grouped_scores))
                                         for question, score in zip(question_group, score_group)])
    # Filter duplicate questions
    sent_idxs, questions, scores = filter_duplicate_questions(sent_idxs, questions, scores)
    return sent_idxs, questions, scores


def generate_questions(input_sents):
    print("Generating questions...")
    input_sents = [" ".join(utils.tokenize_fn(qg_tokenizer, sent)) for sent in input_sents]
    input_sents = [sent if sent else "." for sent in input_sents]  # Program crashes on empty inputs
    scores, gen_qs = qg_model.translate(
        src=[sent.encode() for sent in input_sents[:qg_opt.shard_size]],
        batch_size=qg_opt.batch_size,
        batch_type=qg_opt.batch_type,
    )
    scores = [score[0].item() for score in scores]
    gen_qs = [utils.detokenize_fn(qg_tokenizer, gen_q[0].strip().split(" ")).replace("\n", " ")
              for gen_q in gen_qs]
    print("Done")
    return gen_qs, scores


def answer_questions(input_texts, questions, max_seq_length=384, batch_size=256):
    answers = []
    print("Answering questions...")
    for batch_idx in range(0, len(questions), batch_size):
        batch_answers = document_bert_inference.batch_inference(questions[batch_idx:batch_idx + batch_size],
                                                                input_texts[batch_idx:batch_idx + batch_size],
                                                                qa_model, qa_tokenizer, qa_device,
                                                                max_seq_length, batch_size)
        batch_answers = [utils.strip_answer(answer) for answer in batch_answers]
        answers.extend(batch_answers)
    print("Done")
    return answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run server that generates Q&A pairs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--qg_tokenizer_path", "-qg_tokenizer_path", help="Specify path to HuggingFace GPT2Tokenizer config files.",
                        type=str, required=True)
    parser.add_argument("--qg_model_path", "-qg_model_path", help="Specify path to PyTorch question generation model.",
                        type=str, required=True)
    parser.add_argument("--qa_model_path", "-qa_model_path", help="Specify path to PyTorch question answering model.",
                        type=str, required=True)
    parser.add_argument("--gpu_id", "-gpu_id", help="Specify GPU ID, if available (will default to CPU if not given).",
                        type=int, required=False, default=-1)
    parser.add_argument("--port", "-port", help="Specify port number for server.",
                        type=int, required=False, default=8082)
    args = parser.parse_args()

    qg_tokenizer, qg_opt, qg_model = init_qg_pipeline(args.qg_tokenizer_path, args.qg_model_path, gpu_id=args.gpu_id)
    qa_model, qa_tokenizer, qa_device = init_qa_pipeline(args.qa_model_path)

    app.run(port=args.port, host="0.0.0.0", debug=True)
