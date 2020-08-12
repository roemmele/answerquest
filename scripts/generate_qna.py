import sys
import os
root_dir = os.path.realpath('../')
sys.path.insert(0, root_dir)

import logging
import argparse
import json

from answerquest import QnAPipeline

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'INFO'))


def generate(args):

    with open(args.input_texts_path) as f:
        input_texts = [text.strip() for text in f]

    qna_pipeline = QnAPipeline(qg_tokenizer_path=args.qg_tokenizer_path,
                               qg_model_path=args.qg_model_path,
                               qa_model_path=args.qa_model_path)

    sent_idxs_by_text = []
    questions_by_text = []
    answers_by_text = []

    for idx, text in enumerate(input_texts):
        if text.strip():
            sent_idxs, questions, answers = qna_pipeline.generate_qna_items(
                input_text=text,
                qg_batch_size=args.qg_batch_size,
                qa_batch_size=args.qa_batch_size,
                filter_duplicate_answers=args.filter_duplicate_answers,
                filter_redundant=args.filter_redundant,
                sort_by_sent_order=args.sort_by_sent_order)
        else:
            sent_idxs, questions, answers = [], [], []
        sent_idxs_by_text.append(sent_idxs)
        questions_by_text.append(questions)
        answers_by_text.append(answers)
        if idx and idx % 25 == 0:
            print("generated items for", idx, "texts...")

    with open(args.qna_save_path, 'w') as f:
        json.dump([{'sent_idxs': sent_idxs, 'questions': questions, 'answers': answers}
                   for sent_idxs, questions, answers
                   in zip(sent_idxs_by_text, questions_by_text, answers_by_text)], f)
    print("Saved Q&A items to", args.qna_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lists of Q&A items from input texts.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--qg_tokenizer_path", "-qg_tokenizer_path",
                        help="Specify path of directory containing HuggingFace tokenizer for question generation.",
                        type=str, required=True)
    parser.add_argument("--qg_model_path", "-qg_model_path",
                        help="Specify filepath of question generation model",
                        type=str, required=True)
    parser.add_argument("--qa_model_path", "-qa_model_path",
                        help="Specify directory path of question answering model",
                        type=str, required=True)
    parser.add_argument("--input_texts_path", "-input_texts_path",
                        help="Specify filepath to .txt file containing texts for which Q&A items will be generated (one text per line)",
                        type=str, required=True)
    parser.add_argument("--qna_save_path", "-qna_save_path",
                        help="Specify filepath to which Q&A items will be saved (items saved in .json format).",
                        type=str, required=True)
    parser.add_argument("--qg_batch_size", "-qg_batch_size",
                        help="Specify batch size for question generation model.",
                        type=int, required=False, default=256)
    parser.add_argument("--qa_batch_size", "-qa_batch_size",
                        help="Specify batch size for question answering model.",
                        type=int, required=False, default=256)
    parser.add_argument("--filter_duplicate_answers", "-filter_duplicate_answers",
                        help="Enforce all generated items to have unique answers (for duplicates, only keep item with highest QG score)",
                        action='store_true', default=False)
    parser.add_argument("--filter_redundant", "-filter_redundant",
                        help="Apply a filter that computes token-based Jaccard similarity between all items.\
                        For items with high similarity, only keep the one with the highest QG score.",
                        action='store_true', default=False)
    parser.add_argument("--sort_by_sent_order", "-sort_by_sent_order",
                        help="Return items according to the original order of the sentences they were derived from\
                        (instead of ordered by QG score)",
                        action='store_true', default=False)
    args = parser.parse_args()
    print(vars(args))

    generate(args)
