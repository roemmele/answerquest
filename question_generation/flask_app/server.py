import sys
import json
import numpy
import warnings
warnings.filterwarnings('ignore')

import spacy
import torch
from onmt.bin.translate import translate, _get_parser
from onmt.translate.translator import build_translator
from transformers import GPT2Tokenizer
from flask import Flask, request, session, render_template, jsonify
from question_generation import aux_qa, data_utils


app = Flask(__name__)

spacy_model = spacy.load('en')

# Load QG model components
tokenizer = GPT2Tokenizer.from_pretrained('/home/ec2-user/question_generation/gpt2_tokenizer_vocab/')
parser = _get_parser()
model_filepath = '/home/ec2-user/question_generation/model_step_5600.pt'
qg_batch_size = 256
qg_opt = parser.parse_args(args=['-model', model_filepath,
                                 '-src', "",
                                 '-batch_size', str(qg_batch_size),
                                 '-beam_size', '5',
                                 '-gpu', '0'
                                 ])
qg_top_k = 3
qg_model = build_translator(qg_opt, report_score=True)

qa_batch_size = 256
print("Loaded all model components\n")


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
        input_sents, answers = data_utils.extract_candidate_answers(sent)
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
    input_sents = [" ".join(data_utils.tokenize_fn(tokenizer, sent)) for sent in input_sents]
    input_sents = [sent if sent else "." for sent in input_sents]  # Program crashes on empty inputs
    scores, gen_qs = qg_model.translate(
        src=[sent.encode() for sent in input_sents[:qg_opt.shard_size]],
        batch_size=qg_opt.batch_size,
        batch_type=qg_opt.batch_type,
    )
    scores = [score[0].item() for score in scores]
    gen_qs = [data_utils.detokenize_fn(tokenizer, gen_q[0].strip().split(" ")).replace("\n", " ")
              for gen_q in gen_qs]
    print("Done")
    return gen_qs, scores


def answer_questions(input_texts, questions):
    answers = []
    print("Answering questions...")
    for batch_idx in range(0, len(questions), qa_batch_size):
        batch_answers = [aux_qa.strip_answer(answer) for answer in
                         aux_qa.answer_questions(input_texts[batch_idx:batch_idx + qa_batch_size],
                                                 questions[batch_idx:batch_idx + qa_batch_size])]
        answers.extend(batch_answers)
    print("Done")
    return answers


if __name__ == '__main__':
    app.run(port=8082, host="0.0.0.0", debug=True)
