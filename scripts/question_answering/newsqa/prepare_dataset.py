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
    def __init__(self, tokens, paragraphs, original_consensus_answer):
        self.tokens = tokens
        self.paragraphs = paragraphs
        self.original_consensus_answer = original_consensus_answer


def get_tokens_original_indices(paragraph_obj, tokenizer, start_position, end_position):
    tokens_to_original_index = []
    original_to_tokens_index = []
    whitespace_tokens = paragraph_obj.whitespace_tokens
    tokens = []

    for (i, tok) in enumerate(whitespace_tokens):
        original_to_tokens_index.append(len(tokens))
        subtokens = tokenizer.tokenize(tok)
        for s in subtokens:
            tokens.append(s)
            tokens_to_original_index.append(i)

    if start_position != -1 and end_position != -1:
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


def tokenize_whitespace(whitespace_tokens, tokenizer):
    tokens = []
    for (i, tok) in enumerate(whitespace_tokens):
        subtokens = tokenizer.tokenize(tok)
        for s in subtokens:
            tokens.append(s)
    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", default=None, help="pickle file after preprocessing")
    parser.add_argument("--force_answer", default=1, help="Whether to force the answer to be in the top selection")
    parser.add_argument("--k", default=4, help="Top K paragraphs value")
    parser.add_argument("--output_file_name", default=None, help="train/dev/test")
    args = parser.parse_args()

    docs = pickle.load(open(args.pickle_file, 'rb'))
    k = int(args.k)
    name = args.output_file_name

    ### trial to see if everything is correct
    ##docs = docs[:100]

    bert_model = 'bert-base-uncased'
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    not_in_top_k = 0
    q_p_list = []
    done = 0
    force_answer = int(args.force_answer)
    print ("Total documents: ", len(docs))
    print ("force_answer: ", force_answer)

    less_than_k = 0

    ## needed because documents vary a lot in number of paragraphs and some might not even have 4 paragraphs 
    pad_paragraph = ['[PAD]', '[PAD]']

    for document in docs:
        paragraph_objects = document.paragraphs
        print ("num paragraphs: ", len(paragraph_objects))
        if len(paragraph_objects) < k:
            less_than_k += 1
        paragraphs_whitespace_list = list(map(lambda x: " ".join(x.whitespace_tokens), paragraph_objects))
        paragraphs_tokenized_list = list(map(lambda x: tokenizer.tokenize(x), paragraphs_whitespace_list))
        for question in document.questions:
            tokenized_question = tokenizer.tokenize(question.string)
            ## for this dataset, it may not be topk, so padding is needed
            top_k_objects = return_top_k(paragraphs_tokenized_list, tokenized_question, paragraph_objects, k)
            required_id = question.answer_paragraph_id
            ids = list(map(lambda x: x.id, top_k_objects))
            if force_answer:
                if required_id not in ids:
                    top_k_objects[-1] = paragraph_objects[required_id]
                    not_in_top_k += 1
            top_k_paragraph_answers = []

            #### questions which dont have answers will have all -1, which will need to be fixed while training, so that only one paragraph has 1 one-hot index!!
            for p in top_k_objects:
                if p.id == required_id:
                    top_k_paragraph_answers.append(
                        get_tokens_original_indices(p, tokenizer, question.whitespace_start_index,
                                                    question.whitespace_end_index))
                else:
                    top_k_paragraph_answers.append(get_tokens_original_indices(p, tokenizer, -1, -1))

            ## padding!
            if len(top_k_paragraph_answers) < k:
                for i in range(k - len(top_k_paragraph_answers)):
                    top_k_paragraph_answers.append(ParagraphAnswer(pad_paragraph, -1, -1, None, None))
            q_p_list.append(QuestionParagraph(tokenized_question, top_k_paragraph_answers, question.answer))
        done += 1
        print ('finished doc: ', done)
        print ()
    print ("Length of q_p_list: ", len(q_p_list))
    print ("Paragraphs not in top k: ", not_in_top_k, not_in_top_k / len(q_p_list))
    print ("Padding needed in: ", less_than_k)
    pickle.dump(q_p_list, open(name + '_q_p.pkl', 'wb'))
