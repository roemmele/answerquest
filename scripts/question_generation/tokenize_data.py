import sys
sys.path.append("../../")

import os
import argparse
from transformers import GPT2Tokenizer
from answerquest.utils import tokenize_fn


def tokenize_data(input_texts, questions, tokenizer, tokenize_fn):
    # Add special character after answer annotations so tokenizer will respect space
    tok_input_texts = []
    tok_questions = []
    for idx, (input_text, question) in enumerate(zip(input_texts, questions)):
        tok_question = " ".join(tokenize_fn(tokenizer, question))
        tok_questions.append(tok_question)
        tok_input_text = " ".join(tokenize_fn(tokenizer, input_text))
        tok_input_texts.append(tok_input_text)
        if idx and idx % 10000 == 0:
            print(idx)
    return tok_input_texts, tok_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tokenized dataset for training QG model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tokenizer_path", "-tokenizer_path",
                        help="Path of directory containing adapted HuggingFace tokenizer vocabulary.",
                        type=str, required=True)
    parser.add_argument("--in_data_dir", "-in_data_dir",
                        help="Directory path of data to be tokenized (folder must include questions.txt and input_texts.txt)",
                        type=str, required=True)
    parser.add_argument("--out_data_dir", "-out_data_dir",
                        help="Directory path where tokenized data will be saved.",
                        type=str, required=True)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)

    with open(os.path.join(args.in_data_dir, "input_texts.txt")) as f:
        input_texts = [text.strip() for text in f]

    with open(os.path.join(args.in_data_dir, "questions.txt")) as f:
        questions = [question.strip() for question in f]

    tok_input_texts, tok_questions = tokenize_data(input_texts, questions,
                                                   tokenizer, tokenize_fn=tokenize_fn)

    if not os.path.isdir(args.out_data_dir):
        os.mkdir(args.out_data_dir)

    with open(os.path.join(args.out_data_dir, 'input_texts.txt'), 'w') as f:
        f.write("\n".join(tok_input_texts))
        print("Saved tokenized input texts to", os.path.join(
            args.out_data_dir, 'input_texts.txt'))

    with open(os.path.join(args.out_data_dir, 'questions.txt'), 'w') as f:
        f.write("\n".join(tok_questions))
        print("Saved tokenized questions to", os.path.join(
            args.out_data_dir, 'questions.txt'))
