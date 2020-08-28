import warnings
warnings.filterwarnings('ignore')

import logging
import os
import numpy
import spacy
import en_core_web_sm  # spacy english model
import torch
import re
from onmt.bin.translate import translate, _get_parser
from onmt.translate.translator import build_translator
from transformers import GPT2Tokenizer

from . import utils
from .question_answering import document_bert_inference

torch.multiprocessing.set_start_method('spawn', force=True)
numpy.random.seed(0)
torch.manual_seed(0)
spacy_model = en_core_web_sm.load()
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sort_questions_by_score(sent_idxs, questions, scores):
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
            items_by_question[question] = sent_idx

    sent_idxs, questions = zip(*[(sent_idx, question)
                                 for question, sent_idx
                                 in items_by_question.items()])
    return sent_idxs, questions


def sort_qa_items_by_sent_idx(sent_idxs, questions, answers):
    '''Reorder Q&A pairs according to order of sentences they were originally derived from'''

    sent_idxs_sort_order = numpy.argsort(sent_idxs)
    sent_idxs, questions, answers = zip(*[(sent_idxs[idx], questions[idx], answers[idx])
                                          for idx in sent_idxs_sort_order])
    return sent_idxs, questions, answers


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

    items_by_answer = {}
    for sent_idx, question, answer in zip(sent_idxs, questions, answers):
        if answer not in items_by_answer:
            items_by_answer[answer] = (sent_idx, question)

    (sent_idxs, questions,
     answers) = zip(*[(sent_idx, question, answer)
                      for answer, (sent_idx, question)
                      in items_by_answer.items()])

    return sent_idxs, questions, answers


class QA():

    def __init__(self, model_path):
        (self.model,
         self.tokenizer) = self.init_qa_pipeline(model_path)

    def init_qa_pipeline(self, model_path):
        model, tokenizer, _ = document_bert_inference.load_model(model_path)
        logger.info("Loaded all QA model components")
        return model, tokenizer

    def answer_question(self, input_text, question, max_seq_length=384, top_k=4,
                        is_token=True, max_tokens=300, sliding_size=4,
                        sliding_stride=2, shared_norm=False):
        '''Performs QA for a single question and a single multi-paragraph document (input_text).
        Apply paragraph relevance ranking to get top_k most relevant paragraphs, then
        search these paragraphs specifically for the answers'''

        input_text = input_text.replace("\n", " ")
        input_text = input_text.replace("\r", " ")
        input_text = input_text.replace("\r\n", " ")

        if is_token:
            (original_paragraphs,
             bm25) = document_bert_inference.process_document(input_text,
                                                              self.tokenizer,
                                                              max_tokens)
        else:
            (original_paragraphs,
             bm25) = document_bert_inference.process_sliding_document(input_text,
                                                                      self.tokenizer,
                                                                      sliding_size, sliding_stride)
        obj = document_bert_inference.return_question_paragraph(question, original_paragraphs,
                                                                bm25, self.tokenizer, k=top_k)
        if not shared_norm:
            ans = document_bert_inference.inference(obj, self.model, self.tokenizer,
                                                    device, max_option=True)
        else:
            ans = document_bert_inference.inference(obj, self.model, self.tokenizer,
                                                    device, max_option=False)
        if ans != "":
            return ans
        return "Sorry, the answer cannot be found ....."


class QnAPipeline():

    def __init__(self, qg_tokenizer_path, qg_model_path, qa_model_path):

        (self.qg_tokenizer,
         self.qg_opt,
         self.qg_model) = self.init_qg_pipeline(qg_tokenizer_path, qg_model_path)

        qa = QA(model_path=qa_model_path)
        self.qa_model = qa.model
        self.qa_tokenizer = qa.tokenizer

    def init_qg_pipeline(self, tokenizer_path, model_path, beam_size=5):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        parser = _get_parser()
        opt = parser.parse_args(args=['-model', model_path,
                                      '-src', '',
                                      '-beam_size', str(beam_size),
                                      '-gpu', '0' if device.type == 'cuda' else '-1',
                                      ])
        model = build_translator(opt, report_score=True)
        logger.info("Loaded all QG model components")
        return tokenizer, opt, model

    def init_qa_pipeline(self, model_path):
        model, tokenizer, _ = document_bert_inference.load_model(model_path)
        logger.info("Loaded all QA model components")
        return model, tokenizer

    def generate_qna_items(self, input_text, qg_batch_size=256, qa_batch_size=256,
                           max_qg_input_length=200, filter_duplicate_answers=False,
                           filter_redundant=False, sort_by_sent_order=False):
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
        answers = self.answer_paragraph_level_questions(paragraphs=qa_input_chunks,
                                                        questions=questions,
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

        if filter_duplicate_answers:
            (sent_idxs,
             questions,
             answers) = filter_questions_with_duplicate_answers(sent_idxs,
                                                                questions,
                                                                answers)

        if filter_redundant:
            (sent_idxs,
             questions,
             answers) = filter_redundant_qa_items(sent_idxs,
                                                  questions,
                                                  answers)

        if sort_by_sent_order:
            (sent_idxs,
             questions,
             answers) = sort_qa_items_by_sent_idx(sent_idxs,
                                                  questions,
                                                  answers)

        return sent_idxs, questions, answers

    def get_answer_annotated_input_sents(self, spacy_input_sents, max_input_length=200):
        grouped_input_sents = []  # A group is all annotated sents that map to the same original input sentence
        grouped_answers = []
        for sent in spacy_input_sents:
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
        if not questions:
            return [], []
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

        return sent_idxs, questions

    def generate_questions(self, input_sents, batch_size=256):
        logger.info("Generating questions...")
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
            logger.info("QG complete")
        except:
            logger.info("QG failed on input with {} sentences:".format(len(input_sents)), input_sents)
            gen_qs = []
            scores = []
        return gen_qs, scores

    def answer_paragraph_level_questions(self, paragraphs, questions,
                                         max_seq_length=384, batch_size=256):
        '''Performs QA for several question/paragraph pairs.
        One-to-one relation between each input text (i.e. paragraph) in paragraphs
        and each question in questions. Entire paragraph will be searched for answer to question.
        Does not handle multi-paragraph QA (i.e. no paragraph filtering)'''
        answers = []
        logger.info("Answering questions...")
        for batch_idx in range(0, len(questions), batch_size):
            batch_answers = document_bert_inference.batch_inference(
                questions[batch_idx:batch_idx + batch_size],
                paragraphs[batch_idx:batch_idx + batch_size],
                self.qa_model, self.qa_tokenizer, device,
                max_seq_length, batch_size
            )
            batch_answers = [utils.strip_answer(answer) for answer in batch_answers]
            answers.extend(batch_answers)
        logger.info("QA complete")
        return answers
