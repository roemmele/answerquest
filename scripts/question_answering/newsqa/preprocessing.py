import argparse
import pickle


class Paragraph:
    def __init__(self, id, whitespace_tokens):
        self.id = id
        self.whitespace_tokens = whitespace_tokens


class Question:
    def __init__(self, string):
        self.string = string
        self.answer_paragraph_id = -1
        self.answer = ""
        self.whitespace_start_index = -1
        self.whitespace_end_index = -1


class Document:
    def __init__(self, paragraphs, questions):
        self.paragraphs = paragraphs
        self.questions = questions


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def return_original_index(index, max_seq_length, num_paragraphs):
    for i in range(0, num_paragraphs):
        if index >= max_seq_length * i and index < max_seq_length * (i + 1):
            return (i, index - (max_seq_length * i))


def return_whitespace_tokens(paragraph):
    whitespace_tokens = []
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
    return whitespace_tokens, char_to_word_offset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", default=None, help="newsqa pickled dataset file")
    parser.add_argument("--output_file_name", default=None, help="Either train/dev/test")
    args = parser.parse_args()

    data = pickle.load(open(args.pickle_file, 'rb'))

    chunk_size = 300  ## whitespace chunk tokens for each paragraph

    not_in_same_paragraph = 0

    doc_list = []
    actual_ans = []
    label_ans = []

    for index, document in enumerate(data):
        text = document['text']
        questions = document['questions']
        whitespace_text, char_to_word_offset = return_whitespace_tokens(text)
        paragraph_chunks = []
        paragraph_object_list = []
        question_object_list = []
        ctr = 0
        for i in range(0, len(whitespace_text), chunk_size):
            paragraph_chunks.append(whitespace_text[i:i + chunk_size])
            paragraph_object_list.append(Paragraph(ctr, whitespace_text[i:i + chunk_size]))
            ctr += 1
        for question in questions:
            consensus = question['consensus']
            question_obj = Question(question['q'])
            if 'badQuestion' in consensus or 'noAnswer' in consensus:
                whitespace_s_index = -1
                whitespace_e_index = -1
                paragraph_index = 0
                s_index = -1
                e_index = -1
                question_obj.answer = ""
            else:
                question_obj.answer = text[consensus['s']:consensus['e']]

                whitespace_s_index = char_to_word_offset[consensus['s']]

                ### try converting end consensus to inclusive index
                ### This step is really important!!!
                consensus['e'] = consensus['e'] - 1

                whitespace_e_index = char_to_word_offset[consensus['e']]

                p_s_index, s_index = return_original_index(whitespace_s_index, chunk_size, len(paragraph_chunks))
                p_e_index, e_index = return_original_index(whitespace_e_index, chunk_size, len(paragraph_chunks))

                if p_s_index != p_e_index:
                    paragraph_index = 0
                    s_index = -1
                    e_index = -1
                    not_in_same_paragraph += 1
                else:
                    paragraph_index = p_s_index

            question_obj.answer_paragraph_id = paragraph_index
            question_obj.whitespace_start_index = s_index
            question_obj.whitespace_end_index = e_index  ## there was confusion early on, regarding the inclusive/exclusiveness. Fixing consensus end seems to fix this. Now its like Squad, "inclusive"
            question_object_list.append(question_obj)

            if s_index == -1 and e_index == -1:
                label_ans.append("")
            else:
                label_ans.append(paragraph_chunks[paragraph_index][s_index:e_index + 1])

            actual_ans.append(question_obj.answer)

        doc_list.append(Document(paragraph_object_list, question_object_list))

    print ("Not in same paragraph: ", not_in_same_paragraph)

    pickle.dump(doc_list, open(args.output_file_name + '_doc_list.pkl', 'wb'))
    #pickle.dump(actual_ans, open(args.output_dir_name + '/' + args.output_dir_name + '_actual_ans.pkl', 'wb'))
    #pickle.dump(label_ans, open(args.output_dir_name + '/' + args.output_dir_name + '_label_ans.pkl', 'wb'))
