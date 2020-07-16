import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(root_dir, "question_answering/demo_inference/"))

import warnings
warnings.filterwarnings('ignore')

import numpy
import spacy
import torch
import re
from onmt.bin.translate import translate, _get_parser
from onmt.translate.translator import build_translator
from transformers import GPT2Tokenizer
from gensim.summarization import summarizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from question_generation import utils
from question_answering.demo_inference import document_bert_inference


numpy.random.seed(0)
torch.manual_seed(0)
spacy_model = spacy.load('en')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sort_questions_by_score(sent_idxs, questions, scores):
    # import pdb
    # pdb.set_trace()
    sorted_q_idxs = numpy.argsort(scores)[::-1]
    sent_idxs, questions, scores = zip(*[(sent_idxs[idx], questions[idx], scores[idx])
                                         for idx in sorted_q_idxs])
    return sent_idxs, questions, scores


def filter_duplicate_questions(sent_idxs, questions):
    '''Keep the question duplicate with the highest score.
    Assumes that questions are already sorted by score.'''

    items_by_question = {}
    for sent_idx, question in zip(sent_idxs, questions):
        if question not in items_by_question:
                # or question_score > q_items_by_question[question][1]):
            items_by_question[question] = sent_idx

    sent_idxs, questions = zip(*[(sent_idx, question)
                                 for question, sent_idx
                                 in items_by_question.items()])
    return sent_idxs, questions


def filter_qa_items_by_sent_idx(sent_idxs, questions, answers):
    '''Only return one question-answer item per original input sentence.
    Assumes items are already sorted by QG score, so return first item per sentence.'''

    items_by_sent_idx = {}
    for sent_idx, question, answer in zip(sent_idxs, questions, answers):
        if sent_idx not in items_by_sent_idx:
            items_by_sent_idx[sent_idx] = (question, answer)
    sent_idxs, q_items = zip(*items_by_sent_idx.items())
    questions, answers = zip(*items)
    return sent_idxs, questions, answers


def sort_qa_items_by_sent_idx(sent_idxs, questions, answers):
    '''Reorder Q&A pairs according to order of sentences they were originally derived from'''
    sent_idxs_sort_order = numpy.argsort(sent_idxs)
    sent_idxs, questions, answers = zip(*[(sent_idxs[idx], questions[idx], answers[idx])
                                          for idx in sent_idxs_sort_order])
    return sent_idxs, questions, answers


def filter_qa_items_by_centrality(sent_idxs, questions, answers, max_items=None):

    sent_idxs, questions, answers = list(sent_idxs), list(questions), list(answers)

    selected_sent_idxs = []
    selected_questions = []
    selected_answers = []

    # Convert Q&A pairs to bag-of-words vectors
    vectorizer = CountVectorizer(stop_words='english')
    qa_pair_vectors = vectorizer.fit_transform([question + " " + answer for question, answer
                                                in zip(questions, answers)]).toarray()

    centroid_vector = qa_pair_vectors.sum(axis=0)

    centroid_sim_scores = cosine_similarity(centroid_vector[None],
                                            qa_pair_vectors)[0]

    while questions and (not max_items or len(selected_questions) < max_items):

        centroid_sim_scores = cosine_similarity(centroid_vector[None],
                                                qa_pair_vectors)[0]
        selected_item_idx = numpy.argmax(centroid_sim_scores, axis=0)

        selected_sent_idx = sent_idxs[selected_item_idx]
        selected_sent_idxs.append(selected_sent_idx)

        selected_question = questions[selected_item_idx]
        selected_questions.append(selected_question)

        selected_answer = answers[selected_item_idx]
        selected_answers.append(selected_answer)

        centroid_vector = centroid_vector - qa_pair_vectors[selected_item_idx]

        qa_pair_vectors = numpy.delete(qa_pair_vectors, selected_item_idx, 0)
        sent_idxs.pop(selected_item_idx)
        questions.pop(selected_item_idx)
        answers.pop(selected_item_idx)

    return tuple(selected_sent_idxs), tuple(selected_questions), tuple(selected_answers)


def filter_redundant_qa_items(sent_idxs, questions, answers, score_threshold=0.6):
    '''Filter questions with the same sentence index that have the same answer.
    Assumes questions are ordered by score, so only highest-scoring question will be kept'''
    tokenizer = re.compile(r"(?u)\b[a-zA-Z0-9]+\b").findall
    items_by_sent_idx = {}
    for sent_idx, question, answer in zip(sent_idxs, questions, answers):
        if sent_idx not in items_by_sent_idx:
            items_by_sent_idx[sent_idx] = [(question, answer)]
        else:
            item_is_redundant = False
            for obs_question, obs_answer in items_by_sent_idx[sent_idx]:
                obs_qa_tokens = set(tokenizer(obs_question.lower()) + tokenizer(obs_answer.lower()))
                new_qa_tokens = set(tokenizer(question.lower()) + tokenizer(answer.lower()))
                redundancy_score = (len(obs_qa_tokens.intersection(new_qa_tokens)) /
                                    min(len(obs_qa_tokens), len(new_qa_tokens)))
                if redundancy_score >= score_threshold:
                    item_is_redundant = True
                    break
            if not item_is_redundant:
                items_by_sent_idx[sent_idx].append((question, answer))

    (sent_idxs, questions, answers) = zip(*[(sent_idx, question, answer)
                                            for sent_idx, qas in items_by_sent_idx.items()
                                            for question, answer in qas])

    return sent_idxs, questions, answers


def filter_questions_with_duplicate_answers(sent_idxs, questions, answers):
    '''Filter questions with the same sentence index that have the same answer.
    Assumes questions are ordered by score, so only highest-scoring question will be kept'''
    # import pdb
    # pdb.set_trace()
    items_by_answer = {}
    for sent_idx, question, answer in zip(sent_idxs, questions, answers):
        if answer not in items_by_answer:
            items_by_answer[answer] = (sent_idx, question)

    (sent_idxs, questions,
     answers) = zip(*[(sent_idx, question, answer)
                      for answer, (sent_idx, question)
                      in items_by_answer.items()])

    return sent_idxs, questions, answers


class QnAPipeline():

    def __init__(self, qg_tokenizer_path, qg_model_path, qa_model_path):

        (self.qg_tokenizer,
         self.qg_opt,
         self.qg_model) = self.init_qg_pipeline(qg_tokenizer_path, qg_model_path)

        (self.qa_model,
         self.qa_tokenizer) = self.init_qa_pipeline(qa_model_path)

    def init_qg_pipeline(self, tokenizer_path, model_path, beam_size=5):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        parser = _get_parser()
        opt = parser.parse_args(args=['-model', model_path,
                                      '-src', '',
                                      '-beam_size', str(beam_size),
                                      '-gpu', '0' if device.type == 'cuda' else '-1',
                                      ])
        model = build_translator(opt, report_score=True)
        print("Loaded all QG model components")
        return tokenizer, opt, model

    def init_qa_pipeline(self, model_path):
        model, tokenizer, _ = document_bert_inference.load_model(model_path)
        print("Loaded all QA model components")
        return model, tokenizer  # , device

    def generate_qna_items(self, input_text, qg_batch_size=256,
                           qa_batch_size=256, max_qg_input_length=200):
        spacy_input_sents = list(spacy_model(input_text).sents)

        qg_input_sents = [sent.text for sent in spacy_input_sents]

        (sent_idxs,
         questions) = self.annotate_generate_filter_questions(spacy_input_sents,
                                                              batch_size=qg_batch_size,
                                                              max_input_length=max_qg_input_length)

        # For QA, look for answer to generated question within a 3-sentence context chunk.
        # This context consists of the sentence that generated the question and the sentences
        # immediately before and after this sentence.
        qa_input_chunks = [" ".join(qg_input_sents[max(0, sent_idx - 1):sent_idx + 2])
                           for sent_idx in sent_idxs]
        answers = self.answer_questions(qa_input_chunks, questions,
                                        batch_size=qa_batch_size)
        # Filter questions that don't have answers
        items = [(sent_idx, question, answer) for sent_idx, question, answer
                 in zip(sent_idxs, questions, answers) if answer]
        if not items:  # No answerable questions found
            return [], [], []
        else:
            sent_idxs, questions, answers = zip(*items)

        # Release memory
        torch.cuda.empty_cache()

        assert len(sent_idxs) == len(questions) == len(answers)

        return sent_idxs, questions, answers

    def get_answer_annotated_input_sents(self, spacy_input_sents, max_input_length=200):
        grouped_input_sents = []  # A group is all annotated sents that map to the same original input sentence
        grouped_answers = []
        for sent in spacy_input_sents:
            # print(len(sent))
            # Truncate sentences longer than max_sent_length tokens.
            # The QG model runs out of memory on long segments.
            sent = sent[:max_input_length]
            input_sents, answers = utils.extract_candidate_answers(sent)
            grouped_input_sents.append(input_sents)
            grouped_answers.append(answers)
        return grouped_input_sents, grouped_answers

    def annotate_generate_filter_questions(self, spacy_input_sents,
                                           batch_size=256, max_input_length=200):
        '''Mark all candidate answer spans (currently uses noun chunks and named entities as answer spans)
        in the input sents, which will result in multiple annotated sentences per original input sentence, each having a unqiue
        answer span. Provide all resulting sentences to the QG model. Sort generated questions by score and filter duplicates before returning.'''

        # Get annotated inputs
        grouped_input_sents, _ = self.get_answer_annotated_input_sents(spacy_input_sents,
                                                                       max_input_length)
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
        questions, scores = self.generate_questions(input_sents, batch_size)
        if not questions and not scores:
            return [], [], []
        # Reshape questions/scores into groups
        grouped_questions, grouped_scores = zip(*[(questions[start_idx:end_idx], scores[start_idx:end_idx])
                                                  for start_idx, end_idx in zip(group_boundary_idxs[:-1],
                                                                                group_boundary_idxs[1:])])

        sent_idxs, questions, scores = zip(*[(sent_idx, question, score)
                                             for sent_idx, (question_group, score_group)
                                             in enumerate(zip(grouped_questions, grouped_scores))
                                             for question, score in zip(question_group, score_group)])

        sent_idxs, questions, scores = sort_questions_by_score(sent_idxs,
                                                               questions,
                                                               scores)

        sent_idxs, questions = filter_duplicate_questions(sent_idxs,
                                                          questions)

        assert len(sent_idxs) == len(questions)  # == len(scores)
        return sent_idxs, questions  # , scores

    def generate_questions(self, input_sents, batch_size=256):
        print("Generating questions...")
        input_sents = [" ".join(utils.tokenize_fn(self.qg_tokenizer, sent)) for sent in input_sents]
        input_sents = [sent if sent else "." for sent in input_sents]  # Program crashes on empty inputs
        try:
            scores, gen_qs = self.qg_model.translate(
                src=[sent.encode() for sent in input_sents[:self.qg_opt.shard_size]],
                batch_size=batch_size,
                batch_type='sents',
            )
            scores = [score[0].item() for score in scores]
            gen_qs = [utils.detokenize_fn(self.qg_tokenizer, gen_q[0].strip().split(" ")).replace("\n", " ")
                      for gen_q in gen_qs]
            print("QG complete")
        except:
            print("QG failed on input with {} sentences:".format(len(input_sents)), input_sents)
            gen_qs = []
            scores = []
        return gen_qs, scores

    def answer_questions(self, input_texts, questions, max_seq_length=384, batch_size=256):
        answers = []
        print("Answering questions...")
        for batch_idx in range(0, len(questions), batch_size):
            batch_answers = document_bert_inference.batch_inference(
                questions[batch_idx:batch_idx + batch_size],
                input_texts[batch_idx:batch_idx + batch_size],
                self.qa_model, self.qa_tokenizer, device,
                max_seq_length, batch_size
            )
            batch_answers = [utils.strip_answer(answer) for answer in batch_answers]
            answers.extend(batch_answers)
        print("QA complete")
        return answers
