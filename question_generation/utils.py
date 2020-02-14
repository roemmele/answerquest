import re
import string
import numpy
from collections import defaultdict, OrderedDict, Counter

ne_tagger = None
spacy_model = None
ENTITY_TAGS = {'PERCENT', 'DATE', 'LANGUAGE', 'QUANTITY', 'LAW', 'NORP', 'TIME', 'PRODUCT',
               'PERSON', 'CARDINAL', 'MONEY', 'WORK_OF_ART', 'LOCATION', 'ORGANIZATION', 'EVENT', 'ORDINAL'}


def init_spacy(disabled_pipes=['parser', 'tagger', 'ner']):
    import spacy
    global spacy_model
    spacy_model = spacy.load('en_core_web_sm', disable=disabled_pipes)


def init_NER():
    import tensorflow as tf
    from pyner.neuralner import NER
    tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
    tf.reset_default_graph()

    model_dir = '/home/ec2-user/ca_ner_model/single_model'
    data_dir = '/home/ec2-user/ca_ner_model/single_dataset'
    # model_dir = '/build2/ca/ner/eng/NEURAL_NER/SINGLE_CHARS_MODEL/single_model'
    # data_dir = '/build2/ca/ner/eng/NEURAL_NER/SINGLE_CHARS_MODEL/single_dataset'
    restore_dir = 'best_weights'
    mode = 'test'
    nthreads = 0

    global ne_tagger
    ne_tagger = NER(mode, model_dir, data_dir, restore_dir, nthreads)


def get_chunk(char_idxs, sent, space, tag):
    tag = list(map(lambda x: x.decode('utf-8'), tag))
    chunk_list = []
    i = 0
    while i < len(tag):
        if tag[i][0] in ['S', 'O']:
            if tag[i] != 'O':
                chunk_list.append((char_idxs[i], sent[i] + " " * int(space[i]), tag[i][2:]))
            else:
                chunk_list.append((char_idxs[i], sent[i] + " " * int(space[i]), 'O'))
            i += 1
        else:
            start = i
            end = i + 1
            cur_tag = tag[i][2:]

            while end < len(tag) and tag[end][0] == 'I' and tag[end][2:] == cur_tag:
                end += 1
            if end < len(tag) and tag[start][0] == 'B' and tag[end][0] == 'E':
                final_string = ""
                for k in range(start, end + 1):
                    final_string += sent[k] + " " * int(space[k])
                chunk_list.append((char_idxs[start], final_string, tag[end][2:]))
                i = end + 1
            else:
                for j in range(start, end):
                    chunk_list.append((char_idxs[start], sent[j] + " " * int(space[j]), 'O'))
                i = end
    return chunk_list


def segment_sents(text):
    '''Return sentence start indices in original text in addition to sentences'''
    if not spacy_model or 'parser' not in spacy_model.pipe_names:
        init_spacy(disabled_pipes=['ner'])
    sents = list(spacy_model(text).sents)
    sent_char_start_idxs = [sent[0].idx for sent in sents]
    sents = [sent.text_with_ws for sent in sents]
    return {"sents": sents, "sent_char_start_idxs": sent_char_start_idxs}


def tokenize(sent):
    tokens = [(token.idx, token.text, len(token.whitespace_))
              for token in spacy_model(sent) if token.text.strip()]
    return tokens


def get_tokens_to_tags(input_texts, questions=None, chunk_size=512):
    from nltk import sent_tokenize

    if not spacy_model or 'parser' in spacy_model.pipe_names:  # Reload spacy model without parsing pipe to speed up tokenization
        init_spacy()
    if not ne_tagger:
        init_NER()

    if not questions:
        sents_by_text = [sent_tokenize(input_text.replace("<ANSWER> ", "").replace(" </ANSWER>", ""))
                         for input_text in input_texts]
    else:
        sents_by_text = [sent_tokenize(input_text.replace("<ANSWER> ", "").replace(" </ANSWER>", "")) + [question]
                         for input_text, question in zip(input_texts, questions)]
    print("segmented texts...")
    n_sents_by_text = [len(sents) for sents in sents_by_text]
    text_boundary_idxs = [sum(n_sents_by_text[:idx])
                          for idx in range(len(n_sents_by_text) + 1)]
    sents = [sent for sents in sents_by_text for sent in sents]

    # Give tokenized sentences as input to tagger (preserve space information to reconstruct string below)
    sents = [tokenize(sent) for sent in sents]
    sent_idxs = [[token[0] for token in sent] for sent in sents]
    sent_tokens = [[token[1] for token in sent] for sent in sents]
    sent_spaces = [[token[2] for token in sent] for sent in sents]
    print("tokenized texts...")

    sent_tags = []
    for chunk_idx in range(0, len(sents), chunk_size):
        chunk_sent_tags = ne_tagger.inference([" ".join(tokens)
                                               for tokens in sent_tokens[chunk_idx:chunk_idx + chunk_size]])
        sent_tags.extend(chunk_sent_tags)
        print(chunk_idx)

    # Reshape sents by text
    sent_tags_by_text = [sent_tags[start_idx:end_idx] for start_idx, end_idx
                         in zip(text_boundary_idxs[:-1], text_boundary_idxs[1:])]
    sent_idxs_by_text = [sent_idxs[start_idx:end_idx] for start_idx, end_idx
                         in zip(text_boundary_idxs[:-1], text_boundary_idxs[1:])]
    sent_tokens_by_text = [sent_tokens[start_idx:end_idx] for start_idx, end_idx
                           in zip(text_boundary_idxs[:-1], text_boundary_idxs[1:])]
    sent_spaces_by_text = [sent_spaces[start_idx:end_idx] for start_idx, end_idx
                           in zip(text_boundary_idxs[:-1], text_boundary_idxs[1:])]

    tokens_to_tags = []
    for text_sent_idxs, text_sent_tokens, text_sent_spaces, text_sent_tags in zip(sent_idxs_by_text,
                                                                                  sent_tokens_by_text,
                                                                                  sent_spaces_by_text,
                                                                                  sent_tags_by_text):
        text_tokens_to_tags = OrderedDict()
        for idxs, tokens, spaces, tags in zip(text_sent_idxs, text_sent_tokens,
                                              text_sent_spaces, text_sent_tags):
            for _, token, tag in get_chunk(idxs, tokens, spaces, tags):
                token = token.strip()
                if tag != 'O' and token not in tokens_to_tags:
                    text_tokens_to_tags[token] = tag
        tokens_to_tags.append(text_tokens_to_tags)

    return tokens_to_tags


def mask_entity_tokens(input_text, question, tokens_to_tags):
    '''Mask entitities in text and token-entity tag mapping'''
    serial_tokens_to_tags = {}
    masked_question = question
    masked_input_text = input_text
    ent_type_counts = defaultdict(int)
    for token, tag in tokens_to_tags.items():
        if token in input_text or token in question:
            ent_type_counts[tag] += 1
            tag = "<{}_{}>".format(tag, ent_type_counts[tag])
            serial_tokens_to_tags[token] = tag
    for token in sorted(serial_tokens_to_tags.keys(), key=lambda token: len(token), reverse=True):
        tag = serial_tokens_to_tags[token]
        masked_input_text = re.sub("(?<!_)" + re.escape(token), tag, masked_input_text)
        masked_question = re.sub("(?<!_)" + re.escape(token), tag, masked_question)
    return masked_input_text, masked_question, serial_tokens_to_tags


def tokenize_fn(tokenizer, text, handle_entities=False):
    # Add designated marker ("|") after special tokens so subsequent token will have correct whitespace indicator
    text = re.sub("</?[A-Z]+_?[0-9]*>",
                  lambda match: match.group() + "|",
                  text)
    if handle_entities:
        text = re.sub("(<{})(_[0-9]+>)".format("|".join(ENTITY_TAGS)),
                      lambda match: " ".join(match.groups()),
                      text)
        text = re.sub("\s<" + "({})".format("|".join(ENTITY_TAGS)),
                      lambda match: " Ġ" + match.group().strip(),
                      text)
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token != "|"]
    return tokens


def detokenize_fn(tokenizer, tokens):
    text = tokenizer.convert_tokens_to_string(tokens)
    text = text.replace("| ", "")
    text = text.replace("<ANSWER>", " <ANSWER>")
    text = text.replace("</ANSWER>", " </ANSWER>")
    return text


def unmask_fn(text, tokens_to_tags):
    #     import pdb;pdb.set_trace()
    tags_in_text = list(set(re.findall("</?[A-Z]+_?[0-9]*>", text)))
    print("MASKED:", text)
    if not tags_in_text or not tokens_to_tags:  # No masked tags in text, or text has tags but there's no candidates for replacement
        return text
    tags_to_tokens = {tag: token for token, tag in tokens_to_tags.items()}
    for tag in tags_in_text:
        if not tags_to_tokens:
            break
        tag_type = tag.split("_")[0]
        if tag in tags_to_tokens:  # Exact match in entity alias
            text = text.replace(tag, tags_to_tokens[tag])
            del tags_to_tokens[tag]
        else:  # If no exact match in entity alias, look for entity of same type
            token_found = False
            for avail_tag in tags_to_tokens:
                if tag_type == avail_tag.split("_")[0]:
                    text = text.replace(tag, tags_to_tokens[avail_tag])
                    token_found = True
                    break
            if not token_found:  # If no entity of same type, just fill in with another randomly selected entity
                random_tag = random.choice(list(tags_to_tokens.keys()))
                text = text.replace(tag, tags_to_tokens[random_tag])
    print("UNMASKED:", text, "\n")
    return text


def extract_candidate_answers(spacy_sent):
    '''Take spacy-encoded sentence, identify noun chunks and named entities as candidate answers for question generation,
    and return list of candidate answers as well as list of input sentences annotated with candidate answers'''
    noun_chunks = [chunk for chunk in spacy_sent.noun_chunks
                   if not sum([token.is_stop or token.is_punct for token in chunk]) == len(chunk)]
    named_ents = [ent for ent in spacy_sent.ents]
    other_chunks = get_other_phrases(spacy_sent)
    cand_answers = noun_chunks + named_ents + other_chunks
    # Filter duplicates appearing in both noun chunks and named entities
    cand_answers = {cand_answer.start_char: cand_answer for cand_answer in cand_answers}.values()
    cand_answers = {cand_answer.end_char: cand_answer for cand_answer in cand_answers}.values()
    annotated_sents = []
    sent_answers = []
    for cand_answer in cand_answers:
         # Answer start and end indicies are relative to overall text, so adjust to get indices within this sentence
        answer_start_char = cand_answer.start_char - spacy_sent[0].idx
        answer_end_char = cand_answer.end_char - spacy_sent[0].idx
        annotated_sent = spacy_sent.text.strip()
        # Insert answer marker tokens into sentence
        annotated_sent = (annotated_sent[:answer_start_char]
                          + "<ANSWER> "
                          + annotated_sent[answer_start_char:])
        annotated_sent = (annotated_sent[:answer_end_char + len("<ANSWER> ")]
                          + " </ANSWER>"
                          + annotated_sent[answer_end_char + len("<ANSWER> "):])
        annotated_sents.append(annotated_sent)
        sent_answers.append(cand_answer.text.strip())
    return annotated_sents, sent_answers


def get_other_phrases(spacy_sent):
    phrase_labels = set(['xcomp', 'attr', 'prep', 'obj', 'iobj', 'flat',
                         'fixed', 'csubj', 'ccomp', 'acl', 'conj'])
    phrases = [spacy_sent[token.left_edge.i - spacy_sent[0].i:token.right_edge.i - spacy_sent[0].i + 1]
               for token in spacy_sent if token.dep_ in phrase_labels]
    return phrases


def strip_answer(answer):
    '''Remove whitespace and leading/trailing punctuation'''
    answer = answer.strip()
    answer = re.sub('[{}]+$'.format(re.escape('.,!?:;') + '\s'), '',  answer)
    answer = re.sub('^[{}]+'.format(re.escape('.,!?:;') + '\s'), '',  answer)
    return answer


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_precision_recall_f1_answers(pred_answer, correct_answer):
    prediction_tokens = normalize_answer(pred_answer).split()
    ground_truth_tokens = normalize_answer(correct_answer).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate_answers(pred_answers, all_correct_answers):

    precision_scores = []
    recall_scores = []
    f1_scores = []
    for pred_answer, correct_answers in zip(pred_answers, all_correct_answers):
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        # If multiple correct answers given, compared predicted answer to all and take best score
        for correct_answer in correct_answers:
            precision, recall, f1 = get_precision_recall_f1(pred_answer, correct_answer)
            if f1 >= best_f1:
                best_precision = precision
                best_recall = recall
                best_f1 = f1
        precision_scores.append(best_precision)
        recall_scores.append(best_recall)
        f1_scores.append(best_f1)
    mean_precision = numpy.mean(precision_scores)
    mean_recall = numpy.mean(recall_scores)
    mean_f1 = numpy.mean(f1_scores)

    return {'precision': mean_precision, 'recall': mean_recall, 'f1': mean_f1}
