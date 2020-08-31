import torch
from transformers import BertTokenizer
from .model import BertQA
import nltk
from rank_bm25 import BM25Okapi
import argparse

nltk.download('punkt')  # Ensure sentence-segmentation model is downloaded


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


# Function to get the final processed paragraph with reverse mappings
def get_tokens_original_indices(paragraph_string, tokenizer):
    whitespace_tokens = []
    tokens = []
    tokens_to_original_index = []
    paragraph = paragraph_string

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

    for (i, tok) in enumerate(whitespace_tokens):
        subtokens = tokenizer.tokenize(tok)
        for s in subtokens:
            tokens.append(s)
            tokens_to_original_index.append(i)
    return Paragraph(tokens, whitespace_tokens, tokens_to_original_index)


class Paragraph:

    def __init__(self, tokens, whitespace_tokens, tokens_to_original_index):
        self.tokens = tokens
        self.whitespace_tokens = whitespace_tokens
        self.tokens_to_original_index = tokens_to_original_index


class QuestionParagraph:

    def __init__(self, tokens, paragraphs):
        self.tokens = tokens
        self.paragraphs = paragraphs


# Helps just in case the document remains the same
def return_bm25_object(tokenized_paragraphs):
    bm25 = BM25Okapi(tokenized_paragraphs)
    return bm25


# Function to return the top k tokenized paragraphs
def return_top_k(bm25, original_paragraphs, tokenized_question, k):
    return bm25.get_top_n(tokenized_question, original_paragraphs, n=k)


# Funtion to load the model and tokenizer
def load_model(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model_qa = BertQA.from_pretrained(model_dir)
    model_qa.to(device)
    if n_gpu > 1:
        model_qa = torch.nn.DataParallel(model_qa)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    return model_qa, tokenizer, device


# Slightly expensive, process once if document is the same
def process_document(document, tokenizer, paragraph_size=300):
    if isinstance(document, str):
        document = nltk.sent_tokenize(document)
    original_paragraphs = [[]]
    tokenized_paragraphs = [[]]
    for s in document:
        t = tokenizer.tokenize(s)
        if len(tokenized_paragraphs[-1]) + len(t) < paragraph_size:
            tokenized_paragraphs[-1] += t
            original_paragraphs[-1].append(s)
        else:
            original_paragraphs[-1] = " ".join(original_paragraphs[-1])
            original_paragraphs.append([])
            tokenized_paragraphs.append([])
    if isinstance(original_paragraphs[-1], list):
        original_paragraphs[-1] = " ".join(original_paragraphs[-1])
    bm25 = return_bm25_object(tokenized_paragraphs)
    return original_paragraphs, bm25


# sliding window approach with overlapping paragraphs
def process_sliding_document(document, tokenizer, paragraph_size=5, stride_size=2):
    if isinstance(document, str):
        document = nltk.sent_tokenize(document)
    original_paragraphs = []
    tokenized_paragraphs = []
    for i in range(0, len(document), stride_size):
        p = document[i:i + paragraph_size]
        p_string = " ".join(p)
        p_tokens = tokenizer.tokenize(p_string)
        original_paragraphs.append(p_string)
        tokenized_paragraphs.append(p_tokens)
    bm25 = return_bm25_object(tokenized_paragraphs)
    return original_paragraphs, bm25


def process_simple_split(document, tokenizer):
    original_paragraphs = document.split("\n\n")
    tokenized_paragraphs = list(map(lambda x: tokenizer.tokenize(x), original_paragraphs))
    bm25 = return_bm25_object(tokenized_paragraphs)
    return original_paragraphs, bm25


# get the QuestionParagraph object required for tensors
def return_question_paragraph(question, original_paragraphs, bm25, tokenizer, k=4):
    tokenized_question = tokenizer.tokenize(question)
    top_k_paragraphs = return_top_k(bm25, original_paragraphs, tokenized_question, k)
    paragraph_obj_list = []
    for p in top_k_paragraphs:
        paragraph_obj_list.append(get_tokens_original_indices(p, tokenizer))
    return QuestionParagraph(tokenized_question, paragraph_obj_list)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def process_single_pair(tokens_question, tokens_paragraph, max_seq_length, tokenizer):
    _truncate_seq_pair(tokens_paragraph, tokens_question, max_seq_length - 3)
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in tokens_question:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_paragraph:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


# takes QuestionParagraph object as input and outputs torch tensors
def return_tensors(q_p_object, max_seq_length, tokenizer):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for p in q_p_object.paragraphs:
        ids, mask, seg_ids = process_single_pair(q_p_object.tokens, p.tokens, max_seq_length, tokenizer)
        all_input_ids.append(ids)
        all_input_mask.append(mask)
        all_segment_ids.append(seg_ids)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

    return all_input_ids, all_input_mask, all_segment_ids


def return_original_index(index, max_seq_length, num_paragraphs):
    for i in range(0, num_paragraphs):
        if index >= max_seq_length * i and index < max_seq_length * (i + 1):
            return (i, index - (max_seq_length * i))


# multi-paragraph inference where single question has multiple paragraphs
def inference(q_p_object, model_qa, tokenizer, device, max_seq_length=384, max_option=True):
    num_paragraphs_per_question = len(q_p_object.paragraphs)
    input_ids, input_mask, segment_ids = return_tensors(q_p_object, max_seq_length, tokenizer)
    softmax = torch.nn.Softmax(dim=0)
    model_qa.eval()
    input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)
    with torch.no_grad():
        start_logits, end_logits = model_qa(input_ids, input_mask, segment_ids)
    start_logits = start_logits.detach().cpu()
    end_logits = end_logits.detach().cpu()

    start_logit_list = []
    end_logit_list = []
    for p in range(num_paragraphs_per_question):
        start_logit_list.append(start_logits[p])
        end_logit_list.append(end_logits[p])
    top_n_start = torch.topk(softmax(torch.cat(start_logit_list)), num_paragraphs_per_question)
    top_n_end = torch.topk(softmax(torch.cat(end_logit_list)), num_paragraphs_per_question)
    start_values, start_indices = top_n_start.values.tolist(), top_n_start.indices.tolist()
    end_values, end_indices = top_n_end.values.tolist(), top_n_end.indices.tolist()
    predictions = []
    for i in range(len(start_values)):
        s_p, s_index = return_original_index(start_indices[i], max_seq_length, num_paragraphs_per_question)
        e_p, e_index = return_original_index(end_indices[i], max_seq_length, num_paragraphs_per_question)
        prediction_ans = ""
        new_s_index = s_index - len(q_p_object.tokens) - 2
        new_e_index = e_index - len(q_p_object.tokens) - 2
        if s_p == e_p:
            p = q_p_object.paragraphs[s_p].tokens
            if new_s_index < 0 or new_s_index > len(p) - 1 or new_e_index < 0 or new_e_index > len(p) - 1:
                prediction_ans = ""
            else:
                orig_p = q_p_object.paragraphs[s_p].whitespace_tokens
                orig_s_index = q_p_object.paragraphs[s_p].tokens_to_original_index[new_s_index]
                orig_e_index = q_p_object.paragraphs[s_p].tokens_to_original_index[new_e_index]
                prediction_ans = " ".join(orig_p[orig_s_index:orig_e_index + 1])
        predictions.append((prediction_ans, start_values[i] / 2.0 + end_values[i] / 2.0))
    prediction = ""
    if not max_option:
        prediction = predictions[0][0]
    else:
        for p in predictions:
            if p[0] == "":
                continue
            else:
                prediction = p[0]
                break
    return prediction


# single paragraph-question inference but done batchwise
def batch_inference(question_list, paragraph_list, model_qa, tokenizer, device, max_seq_length, batch_size=256):
    answers = []
    log_softmax = torch.nn.LogSoftmax(dim=1)
    model_qa.eval()

    for i in range(0, len(question_list), batch_size):

        question_batch = question_list[i:i + batch_size]
        paragraph_batch = paragraph_list[i:i + batch_size]
        batch_input_ids = []
        batch_input_mask = []
        batch_segment_ids = []
        batch_paragraph_objects = []
        batch_tokenized_questions = []

        for j in range(len(question_batch)):
            p_obj = get_tokens_original_indices(paragraph_batch[j], tokenizer)
            tokenized_q = tokenizer.tokenize(question_batch[j])
            ids, mask, seg_ids = process_single_pair(tokenized_q, p_obj.tokens, max_seq_length, tokenizer)
            batch_input_ids.append(ids)
            batch_input_mask.append(mask)
            batch_segment_ids.append(seg_ids)
            batch_paragraph_objects.append(p_obj)
            batch_tokenized_questions.append(tokenized_q)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(batch_input_mask, dtype=torch.long)
        batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long)
        batch_input_ids, batch_input_mask, batch_segment_ids = batch_input_ids.to(device), batch_input_mask.to(
            device), batch_segment_ids.to(device)

        with torch.no_grad():
            start_logits, end_logits = model_qa(batch_input_ids, batch_input_mask, batch_segment_ids)

        start_logits = start_logits.detach().cpu()
        end_logits = end_logits.detach().cpu()
        start_softmax = log_softmax(start_logits)
        end_softmax = log_softmax(end_logits)
        predicted_start_indices = torch.argmax(start_softmax, dim=1).tolist()
        predicted_end_indices = torch.argmax(end_softmax, dim=1).tolist()

        for j in range(len(batch_tokenized_questions)):
            new_s_index = predicted_start_indices[j] - len(batch_tokenized_questions[j]) - 2
            new_e_index = predicted_end_indices[j] - len(batch_tokenized_questions[j]) - 2
            p = batch_paragraph_objects[j].tokens
            if new_s_index < 0 or new_s_index > len(p) - 1 or new_e_index < 0 or new_e_index > len(p) - 1:
                ans = ""
            else:
                orig_p = batch_paragraph_objects[j].whitespace_tokens
                orig_s_index = batch_paragraph_objects[j].tokens_to_original_index[new_s_index]
                orig_e_index = batch_paragraph_objects[j].tokens_to_original_index[new_e_index]
                ans = " ".join(orig_p[orig_s_index:orig_e_index + 1])
            answers.append(ans)

    return answers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--document", default=None, help="Document file")
    parser.add_argument("--questions", default=None, help="Questions file")
    parser.add_argument("--model_dir", default=None, help="Directory containing the trained model checkpoint")
    args = parser.parse_args()
    print("Start Inference...")

    model, tokenizer, device = load_model(args.model_dir)

    print("Model Loaded...")

    document = open(args.document).read()
    # document = document.replace("\n", " ")

    ##document = open(args.document).readlines()
    ##document = list(map(lambda x:x.strip('\n'), document))

    questions = open(args.questions).readlines()
    questions = list(map(lambda x: x.strip('\n'), questions))

    original_paragraphs, bm25 = process_document(document, tokenizer, 350)

    ##original_paragraphs, bm25 = process_sliding_document(document, tokenizer, 4, 2)

    ##original_paragraphs, bm25 = process_sliding_document(document, tokenizer, 5, 5)

    ##original_paragraphs, bm25 = process_simple_split(document, tokenizer)

    print("Document processed...")
    print()

    for q in questions:
        obj = return_question_paragraph(q, original_paragraphs, bm25, tokenizer, k=5)
        print('Question: ', q)
        ans = inference(obj, model, tokenizer, device)
        print('Answer: ', ans)
        print()
