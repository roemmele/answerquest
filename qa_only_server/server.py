import sys
import os
root_dir = os.path.realpath('../')
sys.path.insert(0, root_dir)

import argparse
from flask import Flask, request, jsonify, render_template
from answerquest.question_answering import document_bert_inference as br
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MAX_SEQ_LENGTH = 384
top_k = 4
is_token = True
max_tokens = 300
sliding_size = 4
sliding_stride = 2
shared_norm = False
model, tokenizer, device = None, None, None


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', user_image='templates/sdl.png')


@app.route('/documentsqa', methods=['POST'])
def document_predict():
    data = request.get_json(force=True)

    document = data['paragraph']
    document = document.replace("\n", " ")
    document = document.replace("\r", " ")
    document = document.replace("\r\n", " ")

    question = data['question']

    if is_token:
        original_paragraphs, bm25 = br.process_document(document, tokenizer, max_tokens)
    else:
        original_paragraphs, bm25 = br.process_sliding_document(document, tokenizer, sliding_size, sliding_stride)
    obj = br.return_question_paragraph(question, original_paragraphs, bm25, tokenizer, k=top_k)
    if not shared_norm:
        ans = br.inference(obj, model, tokenizer, device, max_option=True)
    else:
        ans = br.inference(obj, model, tokenizer, device, max_option=False)
    if ans != "":
        return ans
    return "Sorry, the answer cannot be found ....."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run server that answers questions from input text.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", "-model_path",
                        help="Specify path to PyTorch question answering model.",
                        type=str, required=True)
    args = parser.parse_args()
    model, tokenizer, device = br.load_model(args.model_path)
    app.run(port=8080, host="0.0.0.0", debug=True)
