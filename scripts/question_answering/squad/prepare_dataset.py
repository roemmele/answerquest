import pickle
from preprocessing import Document, Question, Paragraph
from rank_bm25 import BM25Okapi
import argparse
from transformers import BertTokenizer


## changed it to n = 15, 1, 5 temporarily
def return_top_k(tokenized_paragraph_list, tokenized_question, original_paragraphs, k=4):
    bm25 = BM25Okapi(tokenized_paragraph_list)
    return bm25.get_top_n(tokenized_question, original_paragraphs, n=k)


class ParagraphAnswer:
    def __init__(self, tokens, start_index, end_index, whitespace_tokens, tokens_to_original_index):
        self.tokens = tokens
        self.start_index = start_index
        self.end_index = end_index
        self.whitespace_tokens = whitespace_tokens
        self.tokens_to_original_index = tokens_to_original_index


class QuestionParagraph:
    def __init__(self, tokens, paragraphs, original_all_answers):
        self.tokens = tokens
        self.paragraphs = paragraphs
        self.original_all_answers = original_all_answers


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def get_tokens_original_indices(paragraph_obj, tokenizer, answer, answer_offset):
    whitespace_tokens = []
    tokens = []
    tokens_to_original_index = []
    original_to_tokens_index = []
    paragraph = paragraph_obj.string
    char_to_word_offset = []

    prev_is_whitespace = True
    for c in paragraph:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                whitespace_tokens.append(c)
            else:
                whitespace_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(whitespace_tokens) - 1)

    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + len(answer) - 1]

    for (i, tok) in enumerate(whitespace_tokens):
        original_to_tokens_index.append(len(tokens))
        subtokens = tokenizer.tokenize(tok)
        for s in subtokens:
            tokens.append(s)
            tokens_to_original_index.append(i)

    if answer_offset != -1:
        token_start_position = original_to_tokens_index[start_position]
        if end_position < len(whitespace_tokens) - 1:
            token_end_position = original_to_tokens_index[end_position + 1] - 1
        else:
            token_end_position = len(tokens) - 1
    else:
        token_start_position = -1
        token_end_position = -1

    obj = ParagraphAnswer(tokens, token_start_position, token_end_position, whitespace_tokens, tokens_to_original_index)
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", default=None, help="pickle file after preprocessing")
    parser.add_argument("--force_answer", default=1, help="Wheather to force the answer to be in the top selection")
    parser.add_argument("--k", default=4, help="The top k value for the paragraphs")
    parser.add_argument("--output_file_name", default=None, help="Name of the output file")
    args = parser.parse_args()

    docs = pickle.load(open(args.pickle_file, 'rb'))
    k = int(args.k)
    name = args.output_file_name

    ## for testing only
    ##docs = docs[:2]

    bert_model = 'bert-base-uncased'
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    not_in_top_k = 0
    q_p_list = []
    done = 0
    force_answer = int(args.force_answer)
    print ("Total documents: ", len(docs))
    print ("force_answer: ", force_answer)

    for document in docs:
        paragraph_objects = document.paragraphs
        paragraphs_string_list = list(map(lambda x: x.string, paragraph_objects))
        paragraphs_tokenized_list = list(map(lambda x: tokenizer.tokenize(x), paragraphs_string_list))
        for question in document.questions:
            tokenized_question = tokenizer.tokenize(question.string)
            top_k_objects = return_top_k(paragraphs_tokenized_list, tokenized_question, paragraph_objects, k)
            required_id = question.answer_paragraph_id
            ids = list(map(lambda x: x.id, top_k_objects))
            if force_answer:
                if required_id not in ids:
                    top_k_objects[-1] = paragraph_objects[required_id]
                    not_in_top_k += 1
            top_k_paragraph_answers = []
            for p in top_k_objects:
                if p.id == required_id:
                    top_k_paragraph_answers.append(
                        get_tokens_original_indices(p, tokenizer, question.answer, question.answer_offset))
                else:
                    top_k_paragraph_answers.append(get_tokens_original_indices(p, tokenizer, "", -1))
            q_p_list.append(
                QuestionParagraph(tokenized_question, top_k_paragraph_answers, question.original_all_answers))
        done += 1
        print ('done: ', done)
    print ("Dataset size: ", len(q_p_list))
    print ("Paragraphs not in top 4: ", not_in_top_k, not_in_top_k / len(q_p_list))
    pickle.dump(q_p_list, open(name + '_q_p.pkl', 'wb'))
