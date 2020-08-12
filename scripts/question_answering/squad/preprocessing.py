import json
import argparse
import pickle


class Paragraph:
    def __init__(self, id, string):
        self.id = id
        self.string = string


class Question:
    def __init__(self, string):
        self.string = string
        self.answer_paragraph_id = -1
        self.answer = ""
        self.answer_offset = -1
        self.original_all_answers = []


class Document:
    def __init__(self):
        self.paragraphs = []
        self.questions = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", default=None, help="squad json dataset file")
    parser.add_argument("--output_file_name", default=None, help="name of the output file, train/dev")
    args = parser.parse_args()
    name = args.output_file_name #'dev'
    with open(args.json_file) as j:
        documents = json.load(j)['data']

    ## list of document objects
    doc_list = []
    for d_index in range(len(documents)):
        document_obj = Document()
        for p_index, paragraph in enumerate(documents[d_index]['paragraphs']):
            current_paragraph_text = paragraph['context']
            current_questions = paragraph['qas']
            for q in current_questions:
                qas_id = q["id"]
                question_text = q["question"]
                is_impossible = False
                if "is_impossible" in q:
                    is_impossible = q["is_impossible"]
                if not is_impossible:
                    answer = q["answers"][0]
                    orig_answer_text = answer["text"]
                    orig_all_answers = list(map(lambda x: x["text"], q['answers']))
                    answer_offset = answer["answer_start"]
                else:
                    orig_answer_text = ""
                    orig_all_answers = [""]
                    answer_offset = -1
                question_obj = Question(question_text)
                question_obj.answer_paragraph_id = p_index
                question_obj.answer = orig_answer_text
                question_obj.answer_offset = answer_offset
                question_obj.original_all_answers = orig_all_answers
                document_obj.questions.append(question_obj)
            paragraph_obj = Paragraph(p_index, current_paragraph_text)
            document_obj.paragraphs.append(paragraph_obj)
        doc_list.append(document_obj)
    pickle.dump(doc_list, open(name + '_doc_list.pkl', 'wb'))
