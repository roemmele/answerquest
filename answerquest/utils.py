import re


def tokenize_fn(tokenizer, text):
    # Add designated marker ("|") after special tokens so subsequent token will have correct whitespace indicator
    text = re.sub("</?[A-Z]+_?[0-9]*>",
                  lambda match: match.group() + "|",
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
    annotated_sents_with_answers = {}
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
        if annotated_sent not in annotated_sents_with_answers:
            annotated_sents_with_answers[annotated_sent] = cand_answer.text.strip()
    if annotated_sents_with_answers:
        annotated_sents, sent_answers = zip(*annotated_sents_with_answers.items())
    else:
        annotated_sents, sent_answers = (), ()
    return list(annotated_sents), list(sent_answers)


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
