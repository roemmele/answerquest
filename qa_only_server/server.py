import sys
import os
root_dir = os.path.realpath('../')
sys.path.insert(0, root_dir)

import argparse
from flask import Flask, request, jsonify, render_template
from answerquest import QA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

qa = None

MAX_SEQ_LENGTH = 384
top_k = 4
is_token = True
max_tokens = 300
sliding_size = 4
sliding_stride = 2
shared_norm = False


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', user_image='templates/sdl.png')


@app.route('/documentsqa', methods=['POST'])
def document_predict():
    data = request.get_json(force=True)
    document = data['paragraph']
    question = data['question']
    answer = qa.answer_question(input_text=document,
                                question=question,
                                max_seq_length=MAX_SEQ_LENGTH,
                                top_k=top_k,
                                is_token=is_token,
                                max_tokens=max_tokens,
                                sliding_size=sliding_size,
                                sliding_stride=sliding_stride,
                                shared_norm=shared_norm)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run server that answers questions from input text.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", "-model_path",
                        help="Specify path to PyTorch question answering model.",
                        type=str, required=True)
    args = parser.parse_args()
    qa = QA(args.model_path)
    app.run(port=8080, host="0.0.0.0", debug=True)
