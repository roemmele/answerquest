import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, root_dir)

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
from flask import Flask, request, render_template, jsonify

from question_generation.pipeline import (QnAPipeline, filter_questions_with_duplicate_answers,
                                          filter_redundant_qa_items, sort_qa_items_by_sent_idx)

app = Flask(__name__)

qna_pipeline = None
filter_duplicate_answers = True
filter_redundant_items = True
sort_by_sent_order = True


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/qg_with_qa', methods=['POST'])
def serve_qna_items():
    try:
        input_text = request.values['input_text']
        sent_idxs, questions, answers = qna_pipeline.generate_qna_items(input_text)

        if filter_duplicate_answers:
            (sent_idxs,
             questions,
             answers) = filter_questions_with_duplicate_answers(sent_idxs,
                                                                questions,
                                                                answers)

        if filter_redundant_items:
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

    except Exception as e:
        print(e)
        questions = ['''An error occurred. Perhaps either no Q&A pairs found because text is too short, \
                    or a memory error occurred because the text is too long. Try using a different text.''']
        answers = [""]

    return jsonify({'questions': questions, 'answers': answers})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run server that generates Q&A pairs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--qg_tokenizer_path", "-qg_tokenizer_path",
                        help="Specify path to HuggingFace GPT2Tokenizer config files.",
                        type=str, required=True)
    parser.add_argument("--qg_model_path", "-qg_model_path",
                        help="Specify path to PyTorch question generation model.",
                        type=str, required=True)
    parser.add_argument("--qa_model_path", "-qa_model_path",
                        help="Specify path to PyTorch question answering model.",
                        type=str, required=True)
    parser.add_argument("--port", "-port",
                        help="Specify port number for server.",
                        type=int, required=False, default=8082)
    args = parser.parse_args()

    qna_pipeline = QnAPipeline(args.qg_tokenizer_path,
                               args.qg_model_path,
                               args.qa_model_path)

    app.run(port=args.port, host="0.0.0.0", debug=True)
